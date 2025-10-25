"""
Multi-Iteration Trajectory Generator

Generates trajectories with proper CALL/ASK/ANSWER decision types.
Creates multiple training examples per query (one per iteration).

Location: src/generators/trajectory_generator_multi_iter.py
"""

import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.iteration_state import IterationState, StateManager, ToolResult
from src.generators.decision_engine import DecisionEngine, DecisionType

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """
    One training example in the format: {Qi, COTi, Tool Set i, Decisioni}
    """
    query: str
    chain_of_thought: str
    tool_set: List[Dict[str, Any]]
    decision: str
    context: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self, field_names: Dict[str, str]) -> Dict[str, Any]:
        """
        Convert to dictionary using configured field names.
        
        Args:
            field_names: Dictionary mapping standard names to config names
                        {"query": "Qi", "cot": "COTi", ...}
        """
        result = {
            field_names.get("query", "Qi"): self.query,
            field_names.get("cot", "COTi"): self.chain_of_thought,
            field_names.get("tools", "Tool Set i"): self.tool_set,
            field_names.get("decision", "Decisioni"): self.decision
        }
        
        if self.context:
            result["Context"] = self.context
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


class TrajectoryGeneratorMultiIter:
    """
    Generates multi-iteration trajectories with CALL/ASK/ANSWER decisions.
    
    Each query can generate multiple training examples (one per iteration).
    
    Example:
        Iteration 0: CALL decision → Training Example 1
        Iteration 1: ANSWER decision → Training Example 2
        
    Total: 2 training examples from 1 query
    """
    
    def __init__(
        self,
        bedrock_provider: Any,
        config: Any,
        max_iterations: int = 3,
        use_mock_tools: bool = True
    ):
        """
        Initialize multi-iteration trajectory generator.
        
        Args:
            bedrock_provider: BedrockProvider instance
            config: Configuration object
            max_iterations: Maximum iterations allowed per query
            use_mock_tools: If True, use mocked tool execution (for testing)
        """
        self.provider = bedrock_provider
        self.config = config
        self.max_iterations = max_iterations
        self.use_mock_tools = use_mock_tools
        
        # Initialize decision engine and state manager
        self.decision_engine = DecisionEngine(bedrock_provider)
        self.state_manager = StateManager()
        
        # Load tool definitions
        self.tools = self._load_tool_definitions()
        
        # Get output schema from config
        self.output_schema = config.output.schema
        self.field_names = {
            "query": self.output_schema.fields.query,
            "cot": self.output_schema.fields.cot,
            "tools": self.output_schema.fields.tools,
            "decision": self.output_schema.fields.decision
        }
        
        logger.info(
            f"Initialized TrajectoryGeneratorMultiIter "
            f"(max_iterations={max_iterations}, mock_tools={use_mock_tools})"
        )
    
    def _load_tool_definitions(self) -> List[Dict[str, Any]]:
        """Load tool definitions from tools.json."""
        tools_file = Path(self.config.tools.definitions_file)
        
        if not tools_file.exists():
            logger.warning(f"Tool definitions file not found: {tools_file}")
            return []
        
        try:
            with open(tools_file, 'r') as f:
                tools_data = json.load(f)
                tools = tools_data.get("tools", [])
                logger.info(f"Loaded {len(tools)} tool definitions")
                return tools
        except Exception as e:
            logger.error(f"Error loading tool definitions: {e}")
            return []
    
    def generate_trajectory(
        self,
        query: str,
        query_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TrainingExample]:
        """
        Generate multi-iteration trajectory for a query.
        
        Returns list of training examples (one per iteration).
        
        Args:
            query: User query
            query_id: Optional unique identifier
            metadata: Optional metadata to include in examples
            
        Returns:
            List of TrainingExample objects
        """
        query_id = query_id or str(uuid.uuid4())
        training_examples = []
        
        logger.info(f"Generating trajectory for query_id={query_id}")
        logger.info(f"Query: {query[:100]}...")
        
        # Initialize state: S^(0) = [Q]
        state = self.state_manager.initialize(query_id, query)
        
        # Iterate up to max_iterations
        for iteration in range(self.max_iterations):
            logger.info(f"=== Iteration {iteration} ===")
            
            # Get current context
            context = state.to_context()
            
            # Make decision: CALL, ASK, or ANSWER?
            decision = self.decision_engine.decide(
                query=query,
                context=context["tool_results"],
                available_tools=self.tools,
                iteration=iteration,
                max_iterations=self.max_iterations
            )
            
            logger.info(f"Decision: {decision.type.value}")
            
            # Create training example for this iteration
            example = self._create_training_example(
                query=query,
                context=context["tool_results"],
                decision=decision,
                iteration=iteration,
                metadata=metadata
            )
            training_examples.append(example)
            
            # Execute based on decision type
            if decision.type == DecisionType.CALL:
                # Execute tools and add results to state
                tool_results = self._execute_tools(
                    tools=decision.tools,
                    query=query,
                    iteration=iteration
                )
                state.add_tool_results(tool_results)
                logger.info(f"Called {len(decision.tools)} tools, advancing to iteration {state.iteration}")
                
            elif decision.type == DecisionType.ASK:
                # Stop - waiting for user input
                logger.info(f"ASK decision: {decision.clarification[:50]}...")
                break
                
            elif decision.type == DecisionType.ANSWER:
                # Done!
                logger.info(f"ANSWER decision: {decision.answer[:50]}...")
                break
        
        # Cleanup state
        self.state_manager.delete(query_id)
        
        logger.info(f"Generated {len(training_examples)} training examples")
        return training_examples
    
    def _create_training_example(
        self,
        query: str,
        context: List[Dict[str, Any]],
        decision: Any,  # Decision object
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrainingExample:
        """
        Create one training example for current iteration.
        
        Format: {Qi, COTi, Tool Set i, Decisioni}
        """
        # Format tools based on decision type
        if decision.type == DecisionType.CALL:
            tool_set = self._format_tools_for_call(decision.tools)
            decision_str = "CALL"
            
        elif decision.type == DecisionType.ASK:
            tool_set = []
            decision_str = f"ASK: {decision.clarification}"
            
        elif decision.type == DecisionType.ANSWER:
            tool_set = []
            decision_str = f"ANSWER: {decision.answer}"
        else:
            tool_set = []
            decision_str = "UNKNOWN"
        
        # Build metadata
        example_metadata = {
            "iteration": iteration,
            "decision_type": decision.type.value
        }
        if metadata:
            example_metadata.update(metadata)
        
        return TrainingExample(
            query=query,
            chain_of_thought=decision.reasoning,
            tool_set=tool_set,
            decision=decision_str,
            context=context if iteration > 0 else None,
            metadata=example_metadata
        )
    
    def _format_tools_for_call(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """
        Format tool names into proper tool call format.
        
        Includes tool name, parameters schema, and description.
        """
        formatted_tools = []
        
        for tool_name in tool_names:
            # Find tool definition
            tool_def = next((t for t in self.tools if t["name"] == tool_name), None)
            
            if tool_def:
                formatted_tools.append({
                    "name": tool_name,
                    "description": tool_def.get("description", ""),
                    "parameters": tool_def.get("parameters", {})
                })
            else:
                # Tool not found, add minimal info
                formatted_tools.append({
                    "name": tool_name,
                    "description": f"Tool {tool_name}",
                    "parameters": {}
                })
        
        return formatted_tools
    
    def _execute_tools(
        self,
        tools: List[str],
        query: str,
        iteration: int
    ) -> List[ToolResult]:
        """
        Execute specified tools and return results.
        
        If use_mock_tools=True, returns mocked data.
        Otherwise, executes real tools.
        """
        results = []
        
        for tool_name in tools:
            if self.use_mock_tools:
                result = self._mock_tool_execution(tool_name, query)
            else:
                result = self._real_tool_execution(tool_name, query)
            
            tool_result = ToolResult(
                tool_name=tool_name,
                parameters={"query": query} if tool_name == "search_knowledge_base" else {},
                result=result,
                iteration=iteration
            )
            results.append(tool_result)
            logger.debug(f"Executed tool '{tool_name}': {str(result)[:100]}...")
        
        return results
    
    def _mock_tool_execution(self, tool_name: str, query: str) -> Any:
        """
        Mock tool execution for testing.
        
        Returns simulated data based on tool_name.
        """
        if tool_name == "search_knowledge_base":
            return {
                "documents": [
                    {"content": f"Relevant information about {query[:30]}...", "score": 0.92},
                    {"content": f"Additional context for {query[:30]}...", "score": 0.87},
                    {"content": f"Related details about {query[:30]}...", "score": 0.81}
                ],
                "query": query,
                "n_results": 3
            }
        
        elif tool_name == "calculate":
            return {
                "result": 42.0,
                "expression": "simulated calculation",
                "description": "Mock calculation result"
            }
        
        elif tool_name == "compare_data":
            return {
                "comparison": {
                    "option_a": "Analysis of option A",
                    "option_b": "Analysis of option B"
                },
                "recommendation": "Option A is better for your goals"
            }
        
        elif tool_name == "analyze_trend":
            return {
                "trend": "increasing",
                "analysis": "The metric shows upward trend over time",
                "confidence": 0.85
            }
        
        elif tool_name == "get_allocation":
            return {
                "stocks": 62,
                "bonds": 30,
                "cash": 8,
                "total_value": 487650
            }
        
        else:
            return {
                "status": "success",
                "data": f"Mock data for {tool_name}"
            }
    
    def _real_tool_execution(self, tool_name: str, query: str) -> Any:
        """
        Real tool execution (to be implemented later).
        
        For now, falls back to mock execution.
        """
        logger.warning(f"Real tool execution not implemented yet. Using mock for {tool_name}")
        return self._mock_tool_execution(tool_name, query)
    
    def save_training_examples(
        self,
        examples: List[TrainingExample],
        output_file: Path,
        format: str = "jsonl"
    ):
        """
        Save training examples to file.
        
        Args:
            examples: List of TrainingExample objects
            output_file: Output file path
            format: "jsonl" or "json"
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionaries
        examples_dict = [ex.to_dict(self.field_names) for ex in examples]
        
        if format == "jsonl":
            with open(output_file, 'w') as f:
                for ex in examples_dict:
                    f.write(json.dumps(ex) + '\n')
        elif format == "json":
            with open(output_file, 'w') as f:
                json.dump(examples_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(examples)} training examples to {output_file}")


# Example usage
if __name__ == "__main__":
    from src.core.bedrock_provider import BedrockProvider
    from src.utils import load_config, setup_logger
    
    setup_logger("INFO")
    config = load_config()
    
    # Initialize provider
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        region=config.bedrock.region
    )
    
    # Create generator
    generator = TrajectoryGeneratorMultiIter(
        bedrock_provider=provider,
        config=config,
        max_iterations=3,
        use_mock_tools=True  # Use mocked tools for testing
    )
    
    # Test with a query
    print("\n" + "="*80)
    print("TEST: Multi-Iteration Trajectory Generation")
    print("="*80)
    
    query = "What is my current portfolio allocation?"
    print(f"\nQuery: {query}")
    
    # Generate trajectory
    examples = generator.generate_trajectory(query)
    
    print(f"\n✅ Generated {len(examples)} training examples:")
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Iteration: {ex.metadata['iteration']}")
        print(f"  Decision Type: {ex.metadata['decision_type']}")
        print(f"  COT: {ex.chain_of_thought[:80]}...")
        print(f"  Tools: {[t['name'] for t in ex.tool_set]}")
        print(f"  Decision: {ex.decision[:80]}...")
    
    # Save examples
    output_file = Path("data/output/test_multi_iteration_examples.jsonl")
    generator.save_training_examples(examples, output_file, format="jsonl")
    print(f"\n✅ Saved to: {output_file}")