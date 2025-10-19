"""
Trajectory Generator V2 - With Tool Definitions & Custom Output Format

Key Features:
1. ✅ Uses tool definitions from tools.json
2. ✅ Uses config file settings
3. ✅ Outputs in format: {Qi, COTi, Tool Set i, Decisioni}
4. ✅ Abstraction layer for document references
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call with full definition."""
    name: str
    parameters: Dict[str, Any]
    description: str = ""
    result: Optional[str] = None


@dataclass
class Trajectory:
    """Represents a complete reasoning trajectory."""
    query: str              # Qi
    chain_of_thought: str   # COTi
    tool_calls: List[ToolCall]  # Tool Set i
    decision: str           # Decisioni
    metadata: Dict[str, Any]


class TrajectoryGeneratorV2:
    """
    Generates synthetic trajectories using tool definitions.
    
    Output format: {Qi, COTi, Tool Set i, Decisioni}
    """
    
    def __init__(
        self,
        bedrock_provider: Any,
        vector_store: Any,
        config: Any
    ):
        """
        Initialize trajectory generator.
        
        Args:
            bedrock_provider: BedrockProvider instance
            vector_store: VectorStore instance
            config: Configuration object
        """
        self.provider = bedrock_provider
        self.vector_store = vector_store
        self.config = config
        
        # Load tool definitions
        self.tools = self._load_tool_definitions()
        
        # Get output schema configuration
        self.output_schema = config.output.schema
        
        logger.info(f"Initialized TrajectoryGeneratorV2 with {len(self.tools)} tools")
        logger.info(f"Output format: {self.output_schema.type}")
    
    def _load_tool_definitions(self) -> List[Dict[str, Any]]:
        """Load tool definitions from tools.json."""
        tools_file = Path(self.config.tools.definitions_file)
        
        if not tools_file.exists():
            logger.warning(f"Tool definitions file not found: {tools_file}")
            return []
        
        with open(tools_file, 'r') as f:
            data = json.load(f)
        
        tools = data.get('tools', [])
        logger.info(f"Loaded {len(tools)} tool definitions")
        
        return tools
    
    def _get_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool definition by name."""
        for tool in self.tools:
            if tool['name'] == tool_name:
                return tool
        return None
    
    def generate_trajectory(
        self,
        query: str,
        n_results: int = 3,
        abstract: bool = True
    ) -> Trajectory:
        """
        Generate a complete trajectory for a query.
        
        Args:
            query: User query
            n_results: Number of chunks to retrieve
            abstract: Whether to abstract away document references
            
        Returns:
            Trajectory object with {Qi, COTi, Tool Set i, Decisioni}
        """
        logger.info(f"Generating trajectory for query: {query[:100]}...")
        
        # Step 1: Determine which tools to use
        selected_tools = self._select_tools_for_query(query)
        
        # Step 2: Execute tool calls
        tool_calls = []
        
        # Primary tool: search_knowledge_base
        search_tool = self._get_tool_definition("search_knowledge_base")
        if search_tool:
            search_query = self._extract_search_query(query)
            
            # Execute search
            search_results = self.vector_store.query(
                query_text=search_query,
                n_results=n_results
            )
            
            chunks = search_results['documents'][0]
            metadatas = search_results['metadatas'][0]
            
            tool_call = ToolCall(
                name="search_knowledge_base",
                parameters={
                    "query": search_query,
                    "n_results": n_results
                },
                description=search_tool['description'],
                result=f"Retrieved {len(chunks)} relevant documents"
            )
            tool_calls.append(tool_call)
            
            logger.info(f"Executed search_knowledge_base: {len(chunks)} results")
        
        # Step 3: Generate Chain of Thought (COT)
        cot = self._generate_chain_of_thought(
            query=query,
            tool_calls=tool_calls,
            chunks=chunks
        )
        
        # Step 4: Generate internal answer (with document references)
        internal_answer = self._generate_internal_answer(query, chunks)
        
        # Step 5: Abstract away document references
        if abstract:
            final_decision = self._abstract_answer(internal_answer)
            logger.info("Applied abstraction to remove document references")
        else:
            final_decision = internal_answer
        
        # Step 6: Create trajectory
        trajectory = Trajectory(
            query=query,
            chain_of_thought=cot,
            tool_calls=tool_calls,
            decision=final_decision,
            metadata={
                'source_chunks': len(chunks),
                'abstracted': abstract,
                'sources': list(set([m.get('source', 'unknown') for m in metadatas])),
                'tools_used': [tc.name for tc in tool_calls]
            }
        )
        
        logger.info("✅ Trajectory generated successfully")
        return trajectory
    
    def _select_tools_for_query(self, query: str) -> List[str]:
        """
        Determine which tools are relevant for the query.
        
        For now, uses keyword matching. Can be enhanced with LLM selection.
        """
        query_lower = query.lower()
        selected = []
        
        # Always use search as primary tool
        selected.append("search_knowledge_base")
        
        # Add calculate if query involves math
        math_keywords = ['calculate', 'how much', 'total', 'sum', 'difference', 'percentage']
        if any(kw in query_lower for kw in math_keywords):
            selected.append("calculate")
        
        # Add compare if query involves comparison
        compare_keywords = ['compare', 'difference between', 'versus', 'vs', 'better']
        if any(kw in query_lower for kw in compare_keywords):
            selected.append("compare_data")
        
        # Add analyze_trend if query involves trends/patterns
        trend_keywords = ['trend', 'over time', 'change', 'grow', 'increase', 'decrease']
        if any(kw in query_lower for kw in trend_keywords):
            selected.append("analyze_trend")
        
        return selected
    
    def _generate_chain_of_thought(
        self,
        query: str,
        tool_calls: List[ToolCall],
        chunks: List[str]
    ) -> str:
        """
        Generate Chain of Thought reasoning.
        
        Explains the reasoning process: what tools were used and why.
        """
        cot_prompt = f"""Generate a chain of thought explanation for answering this query.

Query: {query}

Tools used:
{chr(10).join([f"- {tc.name}: {tc.description}" for tc in tool_calls])}

Retrieved information: {len(chunks)} relevant documents found

Explain the reasoning process in 2-3 sentences:
1. What information was needed
2. Which tools were used and why
3. How the information helps answer the query

Chain of thought:"""
        
        cot = self.provider.generate_text(
            prompt=cot_prompt,
            max_tokens=200
        )
        
        return cot.strip()
    
    def _generate_internal_answer(
        self,
        query: str,
        chunks: List[str]
    ) -> str:
        """
        Generate answer using context (may contain figure references).
        """
        context = "\n\n".join(chunks)
        
        prompt = f"""You are a helpful assistant answering questions based on provided context.

Context from documents:
{context}

Question: {query}

Instructions:
- Answer accurately using the context
- Include specific data points, percentages, and numbers
- You may reference figures and pages as they appear in context
- Be precise and specific

Answer:"""
        
        answer = self.provider.generate_text(
            prompt=prompt,
            max_tokens=500
        )
        
        return answer.strip()
    
    def _abstract_answer(self, internal_answer: str) -> str:
        """
        Abstract away document-specific references.
        
        Removes: Figure numbers, page numbers, document phrases
        Keeps: All data, concepts, relationships
        """
        abstraction_prompt = f"""Transform this answer to be generic and transferable.

Original answer:
{internal_answer}

Transform by:
1. Remove figure references: "Figure 2 shows" → "Data shows"
2. Remove page references: "on page 6" → remove
3. Remove document phrases: "According to the document" → "Research shows"
4. KEEP all numbers, percentages, data, and concepts
5. Use generic phrases: "Historical data shows", "Evidence indicates"

Output ONLY the abstracted answer:"""
        
        abstracted = self.provider.generate_text(
            prompt=abstraction_prompt,
            max_tokens=600
        )
        
        # Regex cleanup as backup
        abstracted = self._regex_cleanup(abstracted)
        
        return abstracted.strip()
    
    def _regex_cleanup(self, text: str) -> str:
        """Remove any remaining figure/page references with regex."""
        text = re.sub(r'\b[Ff]igure\s+\d+\b', '', text)
        text = re.sub(r'\b[Ff]ig\.\s+\d+\b', '', text)
        text = re.sub(r'\b[Pp]age\s+\d+\b', '', text)
        text = re.sub(r'\bon\s+page\s+\d+\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_search_query(self, query: str) -> str:
        """Extract search query from user query."""
        return query
    
    def generate_batch(
        self,
        queries: List[str],
        n_results: int = 3,
        abstract: bool = True
    ) -> List[Trajectory]:
        """Generate trajectories for multiple queries."""
        logger.info(f"Generating batch of {len(queries)} trajectories...")
        
        trajectories = []
        for i, query in enumerate(queries, 1):
            try:
                trajectory = self.generate_trajectory(
                    query=query,
                    n_results=n_results,
                    abstract=abstract
                )
                trajectories.append(trajectory)
                logger.info(f"Generated trajectory {i}/{len(queries)}")
                
            except Exception as e:
                logger.error(f"Failed to generate trajectory for query '{query[:50]}...': {e}")
                continue
        
        logger.info(f"✅ Generated {len(trajectories)}/{len(queries)} trajectories")
        return trajectories
    
    def trajectory_to_output_format(
        self,
        trajectory: Trajectory,
        include_metadata: bool = None,
        include_tool_results: bool = None
    ) -> Dict[str, Any]:
        """
        Convert trajectory to configured output format.
        
        Uses config.output.schema to determine field names and structure.
        
        Args:
            trajectory: Trajectory object
            include_metadata: Override config setting
            include_tool_results: Override config setting
            
        Returns:
            Dictionary in configured format
        """
        # Get field names from config
        fields = self.output_schema.fields
        
        # Get include options
        if include_metadata is None:
            include_metadata = self.output_schema.include_metadata
        if include_tool_results is None:
            include_tool_results = self.output_schema.include_tool_results
        
        # Build tool set
        tool_set = []
        for tc in trajectory.tool_calls:
            tool_info = {
                "name": tc.name,
                "parameters": tc.parameters,
                "description": tc.description
            }
            if include_tool_results and tc.result:
                tool_info["result"] = tc.result
            tool_set.append(tool_info)
        
        # Build output in configured format
        output = {
            fields.query: trajectory.query,
            fields.cot: trajectory.chain_of_thought,
            fields.tools: tool_set,
            fields.decision: trajectory.decision
        }
        
        # Add metadata if enabled
        if include_metadata:
            output["metadata"] = trajectory.metadata
        
        return output
    
    def save_trajectories(
        self,
        trajectories: List[Trajectory],
        output_path: str,
        format: str = None
    ):
        """
        Save trajectories to file in configured format.
        
        Args:
            trajectories: List of trajectories
            output_path: Output file path
            format: Override config format ("jsonl" or "json")
        """
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use config format if not specified
        if format is None:
            format = self.config.output.format
        
        # Convert trajectories to output format
        output_data = [
            self.trajectory_to_output_format(t)
            for t in trajectories
        ]
        
        # Save based on format
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for item in output_data:
                    f.write(json.dumps(item) + '\n')
        
        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
        
        elif format == "both":
            # Save as JSONL
            jsonl_path = output_path.with_suffix('.jsonl')
            with open(jsonl_path, 'w') as f:
                for item in output_data:
                    f.write(json.dumps(item) + '\n')
            
            # Save as JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"✅ Saved to both {jsonl_path} and {json_path}")
            return
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✅ Saved {len(trajectories)} trajectories to {output_path}")


# Keep the QuestionGenerator as-is (it's still good)
from .trajectory_generator import QuestionGenerator