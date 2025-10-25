"""
Iteration State Management

Manages stateless iteration state: S^(i) = [Q, D_T1, D_T2, ...]

Each iteration, the LLM sees:
- The original query Q
- All accumulated tool results from previous iterations
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """
    Result from a single tool call.
    
    Contains:
    - Which tool was called
    - What parameters were used
    - What data was returned
    - Which iteration it was called in
    """
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    iteration: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool": self.tool_name,
            "parameters": self.parameters,
            "data": self.result,
            "iteration": self.iteration,
            "timestamp": self.timestamp
        }
    
    def __repr__(self) -> str:
        return f"ToolResult(tool={self.tool_name}, iter={self.iteration})"


@dataclass
class IterationState:
    """
    State at iteration i: S^(i) = [Q, Tool_Results]
    
    This is STATELESS - the LLM doesn't remember previous iterations.
    It always sees:
    - The original query
    - ALL accumulated tool results so far
    """
    query: str
    query_id: str
    iteration: int = 0
    tool_results: List[ToolResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_context(self) -> Dict[str, Any]:
        """
        Convert state to context for LLM input.
        
        Returns:
            {
                "query": original query,
                "iteration": current iteration number,
                "tool_results": [list of tool results]
            }
        """
        return {
            "query": self.query,
            "iteration": self.iteration,
            "tool_results": [tr.to_dict() for tr in self.tool_results]
        }
    
    def add_tool_results(self, results: List[ToolResult]):
        """
        Add tool results and advance to next iteration.
        
        Args:
            results: List of ToolResult objects from this iteration
        """
        self.tool_results.extend(results)
        self.iteration += 1
        logger.info(f"Advanced to iteration {self.iteration} (have {len(self.tool_results)} total tool results)")
    
    def get_called_tools(self) -> List[str]:
        """Get list of tool names that have been called."""
        return [tr.tool_name for tr in self.tool_results]
    
    def has_tool_result(self, tool_name: str) -> bool:
        """Check if a specific tool has been called."""
        return tool_name in self.get_called_tools()
    
    def get_tool_result(self, tool_name: str) -> Optional[ToolResult]:
        """Get result from a specific tool (if called)."""
        for tr in self.tool_results:
            if tr.tool_name == tool_name:
                return tr
        return None
    
    def __repr__(self) -> str:
        return (
            f"IterationState("
            f"query_id={self.query_id}, "
            f"iteration={self.iteration}, "
            f"tools_called={len(self.tool_results)})"
        )


class StateManager:
    """
    Manages iteration states for multiple queries.
    
    Each query gets a unique query_id and its own IterationState.
    """
    
    def __init__(self):
        """Initialize state manager."""
        self.states: Dict[str, IterationState] = {}
        logger.info("Initialized StateManager")
    
    def initialize(self, query_id: str, query: str) -> IterationState:
        """
        Initialize state for a new query: S^(0) = [Q]
        
        Args:
            query_id: Unique identifier for this query
            query: The actual query text
            
        Returns:
            IterationState initialized at iteration 0
        """
        state = IterationState(
            query=query,
            query_id=query_id,
            iteration=0,
            tool_results=[]
        )
        
        self.states[query_id] = state
        logger.info(f"Initialized state for query_id={query_id}")
        
        return state
    
    def get(self, query_id: str) -> Optional[IterationState]:
        """
        Get state for a specific query.
        
        Args:
            query_id: Query identifier
            
        Returns:
            IterationState if exists, None otherwise
        """
        return self.states.get(query_id)
    
    def exists(self, query_id: str) -> bool:
        """Check if state exists for a query."""
        return query_id in self.states
    
    def delete(self, query_id: str):
        """Delete state for a query (cleanup)."""
        if query_id in self.states:
            del self.states[query_id]
            logger.info(f"Deleted state for query_id={query_id}")
    
    def clear(self):
        """Clear all states (cleanup)."""
        count = len(self.states)
        self.states.clear()
        logger.info(f"Cleared {count} states")
    
    def get_all_query_ids(self) -> List[str]:
        """Get all query IDs currently tracked."""
        return list(self.states.keys())
    
    def __len__(self) -> int:
        """Return number of queries being tracked."""
        return len(self.states)
    
    def __repr__(self) -> str:
        return f"StateManager(tracking={len(self.states)} queries)"


# Helper functions for common operations

def create_tool_result(
    tool_name: str,
    parameters: Dict[str, Any],
    result: Any,
    iteration: int
) -> ToolResult:
    """
    Convenience function to create a ToolResult.
    
    Args:
        tool_name: Name of the tool
        parameters: Parameters passed to the tool
        result: Data returned by the tool
        iteration: Which iteration this was called in
        
    Returns:
        ToolResult object
    """
    return ToolResult(
        tool_name=tool_name,
        parameters=parameters,
        result=result,
        iteration=iteration
    )


def format_context_for_display(state: IterationState) -> str:
    """
    Format iteration state context for human-readable display.
    
    Args:
        state: IterationState object
        
    Returns:
        Formatted string representation
    """
    lines = [
        f"Query: {state.query}",
        f"Iteration: {state.iteration}",
        f"Tools called: {len(state.tool_results)}",
    ]
    
    if state.tool_results:
        lines.append("\nTool Results:")
        for idx, tr in enumerate(state.tool_results, 1):
            lines.append(f"  {idx}. {tr.tool_name} (iteration {tr.iteration})")
            lines.append(f"     Parameters: {tr.parameters}")
            lines.append(f"     Result: {str(tr.result)[:100]}...")
    
    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    import uuid
    
    print("\n" + "="*80)
    print("TEST: ITERATION STATE MANAGEMENT")
    print("="*80)
    
    # Create state manager
    manager = StateManager()
    print(f"\nCreated: {manager}")
    
    # Initialize state for a query
    query_id = str(uuid.uuid4())
    query = "What is my current portfolio allocation?"
    
    print(f"\nInitializing state for query: '{query}'")
    state = manager.initialize(query_id, query)
    print(f"Initial state: {state}")
    print(f"Context: {state.to_context()}")
    
    # Simulate iteration 0: CALL tools
    print("\n" + "-"*80)
    print("ITERATION 0: Calling tools")
    print("-"*80)
    
    # Create tool results
    results_iter_0 = [
        create_tool_result(
            tool_name="search_knowledge_base",
            parameters={"query": "portfolio allocation", "n_results": 3},
            result={"documents": ["doc1", "doc2", "doc3"]},
            iteration=0
        ),
        create_tool_result(
            tool_name="get_allocation",
            parameters={},
            result={"stocks": 62, "bonds": 30, "cash": 8},
            iteration=0
        )
    ]
    
    # Add results to state
    state.add_tool_results(results_iter_0)
    print(f"State after iteration 0: {state}")
    print(f"Tools called so far: {state.get_called_tools()}")
    
    # Show context for iteration 1
    print("\n" + "-"*80)
    print("ITERATION 1: Context for LLM")
    print("-"*80)
    context = state.to_context()
    print(f"Query: {context['query']}")
    print(f"Iteration: {context['iteration']}")
    print(f"Tool results available:")
    for tr in context['tool_results']:
        print(f"  - {tr['tool']}: {tr['data']}")
    
    # Display formatted context
    print("\n" + "-"*80)
    print("FORMATTED CONTEXT")
    print("-"*80)
    print(format_context_for_display(state))
    
    # Test state retrieval
    print("\n" + "-"*80)
    print("STATE RETRIEVAL")
    print("-"*80)
    retrieved_state = manager.get(query_id)
    print(f"Retrieved state: {retrieved_state}")
    print(f"Has search_knowledge_base? {retrieved_state.has_tool_result('search_knowledge_base')}")
    print(f"Has calculate? {retrieved_state.has_tool_result('calculate')}")
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    print(f"States before cleanup: {len(manager)}")
    manager.delete(query_id)
    print(f"States after cleanup: {len(manager)}")
    
    print("\n✅ IterationState test complete!")