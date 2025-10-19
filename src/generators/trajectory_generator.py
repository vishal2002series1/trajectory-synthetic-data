"""
Trajectory Generator - Creates synthetic training data from documents.

Key Features:
1. Query ChromaDB for relevant context
2. Generate answers grounded in document data
3. Abstract away document-specific references (figures, pages)
4. Output generic, transferable training data
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call in a trajectory."""
    tool: str
    query: str
    results: List[str]


@dataclass
class Trajectory:
    """Represents a complete reasoning trajectory."""
    query: str
    tool_calls: List[ToolCall]
    reasoning: str
    answer: str
    metadata: Dict[str, Any]


class TrajectoryGenerator:
    """Generates synthetic trajectories from document knowledge base."""
    
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
        
        logger.info("Initialized TrajectoryGenerator")
    
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
            Trajectory object
        """
        logger.info(f"Generating trajectory for query: {query[:100]}...")
        
        # Step 1: Retrieve relevant context from ChromaDB
        search_results = self.vector_store.query(
            query_text=query,
            n_results=n_results
        )
        
        # Extract chunks
        chunks = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        
        logger.info(f"Retrieved {len(chunks)} relevant chunks")
        
        # Step 2: Generate answer using context (with figure references)
        internal_answer = self._generate_internal_answer(query, chunks)
        
        # Step 3: Abstract away document-specific references
        if abstract:
            final_answer = self._abstract_answer(internal_answer)
            logger.info("Applied abstraction to remove document references")
        else:
            final_answer = internal_answer
        
        # Step 4: Create trajectory
        tool_calls = [
            ToolCall(
                tool="search_knowledge_base",
                query=self._extract_search_query(query),
                results=chunks
            )
        ]
        
        trajectory = Trajectory(
            query=query,
            tool_calls=tool_calls,
            reasoning=self._generate_reasoning(query, chunks),
            answer=final_answer,
            metadata={
                'source_chunks': len(chunks),
                'abstracted': abstract,
                'sources': [m.get('source', 'unknown') for m in metadatas]
            }
        )
        
        logger.info("✅ Trajectory generated successfully")
        return trajectory
    
    def _generate_internal_answer(
        self,
        query: str,
        chunks: List[str]
    ) -> str:
        """
        Generate answer using context (may contain figure references).
        
        This is the internal step where we USE figure references
        to ground the answer in actual data.
        """
        context = "\n\n".join(chunks)
        
        prompt = f"""You are a helpful assistant answering questions based on provided context.

Context from documents:
{context}

Question: {query}

Instructions:
- Answer the question accurately using the context
- Include specific data points, percentages, and numbers
- You may reference figures and pages as they appear in the context
- Be precise and specific
- If the context doesn't contain enough information, say so

Answer:"""
        
        answer = self.provider.generate(
            prompt=prompt,
            max_tokens=500
        )
        
        return answer.strip()
    
    def _abstract_answer(self, internal_answer: str) -> str:
        """
        Abstract away document-specific references.
        
        This is the CRITICAL step that removes:
        - Figure numbers (e.g., "Figure 2", "Fig. 3")
        - Page numbers (e.g., "Page 6", "on page 12")
        - Document-specific phrases (e.g., "According to the document")
        
        But KEEPS:
        - All factual data and numbers
        - Concepts and relationships
        - Explanations
        """
        abstraction_prompt = f"""You are creating training data for a general-purpose AI assistant.

Original answer (may contain document references):
{internal_answer}

Transform this into a GENERIC answer by:

1. Remove ALL figure references:
   - "Figure 2 shows" → "Data shows" or "Research indicates"
   - "According to Figure 3" → "Historical data demonstrates"
   - "As shown in Figure 5" → "Analysis reveals"

2. Remove ALL page references:
   - "on page 6" → remove entirely
   - "Page 12 indicates" → "Evidence indicates"

3. Remove document-specific phrases:
   - "According to the document" → "Research shows"
   - "The report states" → "Studies indicate"
   - "This document shows" → "Data demonstrates"

4. KEEP all factual information:
   - Percentages, numbers, dates
   - Concepts and relationships
   - Explanations and reasoning

5. Use generic academic phrases:
   - "Historical data shows"
   - "Research indicates"
   - "Studies demonstrate"
   - "Analysis reveals"
   - "Evidence suggests"

Output ONLY the abstracted answer. Do not include any preamble or explanation.

Abstracted answer:"""
        
        abstracted = self.provider.generate(
            prompt=abstraction_prompt,
            max_tokens=600
        )
        
        # Additional cleanup with regex (backup)
        abstracted = self._regex_cleanup(abstracted)
        
        return abstracted.strip()
    
    def _regex_cleanup(self, text: str) -> str:
        """
        Regex-based cleanup as a safety net.
        Remove any remaining figure/page references.
        """
        # Remove figure references
        text = re.sub(r'\b[Ff]igure\s+\d+\b', '', text)
        text = re.sub(r'\b[Ff]ig\.\s+\d+\b', '', text)
        
        # Remove page references
        text = re.sub(r'\b[Pp]age\s+\d+\b', '', text)
        text = re.sub(r'\bon\s+page\s+\d+\b', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _extract_search_query(self, query: str) -> str:
        """
        Extract a good search query from the user query.
        
        This is the query that would be sent to the knowledge base.
        """
        # Simple version: use the query as-is
        # Could be enhanced with keyword extraction
        return query
    
    def _generate_reasoning(
        self,
        query: str,
        chunks: List[str]
    ) -> str:
        """
        Generate reasoning explaining the thought process.
        
        This helps create more instructive training data.
        """
        reasoning_prompt = f"""Given the query and retrieved context, explain the reasoning process.

Query: {query}

Retrieved context available: {len(chunks)} relevant documents

Explain in 1-2 sentences what information was needed and how the context helps answer the query.

Reasoning:"""
        
        reasoning = self.provider.generate(
            prompt=reasoning_prompt,
            max_tokens=150
        )
        
        return reasoning.strip()
    
    def generate_batch(
        self,
        queries: List[str],
        n_results: int = 3,
        abstract: bool = True
    ) -> List[Trajectory]:
        """
        Generate trajectories for multiple queries.
        
        Args:
            queries: List of queries
            n_results: Number of chunks per query
            abstract: Whether to abstract references
            
        Returns:
            List of Trajectory objects
        """
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
    
    def trajectory_to_training_format(
        self,
        trajectory: Trajectory,
        include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Convert trajectory to training format (OpenAI-style messages).
        
        Args:
            trajectory: Trajectory object
            include_reasoning: Whether to include reasoning in assistant message
            
        Returns:
            Training example dict
        """
        # Build assistant content
        assistant_content = trajectory.answer
        
        if include_reasoning:
            assistant_content = f"Reasoning: {trajectory.reasoning}\n\nAnswer: {trajectory.answer}"
        
        training_example = {
            "messages": [
                {
                    "role": "user",
                    "content": trajectory.query
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ],
            "metadata": trajectory.metadata
        }
        
        return training_example
    
    def save_trajectories(
        self,
        trajectories: List[Trajectory],
        output_path: str,
        format: str = "jsonl",
        include_reasoning: bool = False
    ):
        """
        Save trajectories to file.
        
        Args:
            trajectories: List of trajectories
            output_path: Output file path
            format: Output format ("jsonl" or "json")
            include_reasoning: Whether to include reasoning
        """
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        training_examples = [
            self.trajectory_to_training_format(t, include_reasoning)
            for t in trajectories
        ]
        
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for example in training_examples:
                    f.write(json.dumps(example) + '\n')
        
        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(training_examples, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✅ Saved {len(trajectories)} trajectories to {output_path}")


class QuestionGenerator:
    """Generates questions from document chunks."""
    
    def __init__(self, bedrock_provider: Any):
        """
        Initialize question generator.
        
        Args:
            bedrock_provider: BedrockProvider instance
        """
        self.provider = bedrock_provider
        logger.info("Initialized QuestionGenerator")
    
    def generate_questions_from_chunk(
        self,
        chunk: str,
        n_questions: int = 3,
        complexity: str = "medium"
    ) -> List[str]:
        """
        Generate questions that can be answered using the chunk.
        
        Args:
            chunk: Text chunk
            n_questions: Number of questions to generate
            complexity: "simple", "medium", or "complex"
            
        Returns:
            List of questions
        """
        complexity_guidance = {
            "simple": "factual, straightforward questions requiring direct lookup",
            "medium": "questions requiring some interpretation and synthesis",
            "complex": "multi-part questions requiring deep analysis and reasoning"
        }
        
        prompt = f"""Given the following text, generate {n_questions} {complexity} questions that can be answered using this content.

Text:
{chunk[:2000]}

Generate {complexity_guidance.get(complexity, 'medium')} questions.

Requirements:
- Questions should be natural and realistic
- Do NOT reference figure numbers or page numbers in questions
- Ask about concepts, relationships, and data
- Make questions generally applicable, not document-specific

Output format:
1. [Question 1]
2. [Question 2]
3. [Question 3]

Questions:"""
        
        response = self.provider.generate(
            prompt=prompt,
            max_tokens=300
        )
        
        # Parse questions
        questions = self._parse_questions(response)
        
        logger.info(f"Generated {len(questions)} {complexity} questions")
        return questions
    
    def _parse_questions(self, response: str) -> List[str]:
        """Parse questions from numbered list."""
        questions = []
        
        # Split by lines
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Match numbered questions (1. , 2. , etc.)
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                questions.append(match.group(1).strip())
        
        return questions
    
    def generate_questions_batch(
        self,
        chunks: List[str],
        questions_per_chunk: int = 3,
        complexity_distribution: Dict[str, float] = None
    ) -> List[str]:
        """
        Generate questions from multiple chunks.
        
        Args:
            chunks: List of text chunks
            questions_per_chunk: Questions to generate per chunk
            complexity_distribution: Distribution of complexity levels
            
        Returns:
            List of all generated questions
        """
        if complexity_distribution is None:
            complexity_distribution = {
                'simple': 0.3,
                'medium': 0.5,
                'complex': 0.2
            }
        
        all_questions = []
        
        for i, chunk in enumerate(chunks, 1):
            # Determine complexity for this chunk
            import random
            rand = random.random()
            cumsum = 0
            complexity = 'medium'
            
            for comp, prob in complexity_distribution.items():
                cumsum += prob
                if rand <= cumsum:
                    complexity = comp
                    break
            
            try:
                questions = self.generate_questions_from_chunk(
                    chunk=chunk,
                    n_questions=questions_per_chunk,
                    complexity=complexity
                )
                all_questions.extend(questions)
                logger.info(f"Generated questions from chunk {i}/{len(chunks)}")
                
            except Exception as e:
                logger.error(f"Failed to generate questions from chunk {i}: {e}")
                continue
        
        logger.info(f"✅ Generated {len(all_questions)} total questions from {len(chunks)} chunks")
        return all_questions