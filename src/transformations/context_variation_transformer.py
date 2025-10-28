"""
Context Variation Transformer - Stateless Version

Generates training examples with different context states to teach models:
- How to identify available information in context
- How to recognize missing required information  
- How to filter irrelevant information
- How to handle incomplete data

Key Principle: STATELESS REASONING
- Model only sees what's in the Context field
- Uses "Context contains X" not "I called X"
- Each example is independent

Expansion: ×3 per training example
- COMPLETE: All necessary information available (baseline)
- MISSING_CONTEXT: Required information not in context  
- EXTRA_CONTEXT: Irrelevant information added to context
"""

from typing import Dict, List, Optional, Any
import copy
import random

from ..utils import get_logger

logger = get_logger(__name__)


class ContextVariationTransformer:
    """
    Generates context variations for stateless training.
    
    Teaches model to:
    - Identify what information is currently available
    - Recognize what information is missing
    - Filter irrelevant information
    - Make appropriate tool calls based on context gaps
    
    Expansion: ×3 per training example
    """
    
    # Pool of irrelevant tools that can be added as noise
    IRRELEVANT_TOOLS = [
        {
            "tool": "analyze_trend",
            "data": {
                "trend": "increasing",
                "analysis": "The metric shows upward trend over time",
                "confidence": 0.85
            },
            "iteration": 99,
            "timestamp": "2025-01-01T00:00:00.000000"
        },
        {
            "tool": "calculate",
            "data": {
                "result": 42.0,
                "expression": "simulated calculation",
                "description": "Mock calculation result"
            },
            "iteration": 99,
            "timestamp": "2025-01-01T00:00:00.000000"
        },
        {
            "tool": "compare_data",
            "data": {
                "comparison": {
                    "option_a": "Analysis of option A",
                    "option_b": "Analysis of option B"
                },
                "recommendation": "Option A is better for your goals"
            },
            "iteration": 99,
            "timestamp": "2025-01-01T00:00:00.000000"
        },
        {
            "tool": "get_market_data",
            "data": {
                "status": "success",
                "data": "Mock market data for irrelevant context"
            },
            "iteration": 99,
            "timestamp": "2025-01-01T00:00:00.000000"
        }
    ]
    
    def __init__(self):
        """Initialize context variation transformer."""
        self.variation_types = [
            "COMPLETE",
            "MISSING_CONTEXT",
            "EXTRA_CONTEXT"
        ]
        logger.info("Initialized ContextVariationTransformer (stateless, ×3 expansion)")
    
    def transform(
        self,
        training_example: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate context variations from a training example.
        
        Args:
            training_example: Original example with Context field
        
        Returns:
            Dictionary mapping variation type to modified example:
            {
                "COMPLETE": {...},           # Original with all context
                "MISSING_CONTEXT": {...},    # Missing required data
                "EXTRA_CONTEXT": {...}       # Added irrelevant data
            }
        """
        variations = {}
        
        # 1. COMPLETE: Baseline (original)
        variations["COMPLETE"] = self._create_complete_variation(training_example)
        
        # 2. MISSING_CONTEXT: Remove required data
        if self._can_create_missing_variation(training_example):
            variations["MISSING_CONTEXT"] = self._create_missing_variation(training_example)
        
        # 3. EXTRA_CONTEXT: Add irrelevant data
        if self._can_create_extra_variation(training_example):
            variations["EXTRA_CONTEXT"] = self._create_extra_variation(training_example)
        
        logger.debug(f"Generated {len(variations)} context variations for query: {training_example.get('Q', '')[:50]}...")
        return variations
    
    def _create_complete_variation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create COMPLETE variation (baseline with all context).
        
        This is the original example with metadata added.
        """
        complete = copy.deepcopy(example)
        
        # Update metadata
        if "metadata" not in complete:
            complete["metadata"] = {}
        
        complete["metadata"]["context_variation"] = "COMPLETE"
        complete["metadata"]["variation_description"] = "All necessary information available in context"
        complete["metadata"]["expected_behavior"] = "Model has all needed data and can answer/decide directly"
        
        return complete
    
    def _create_missing_variation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create MISSING_CONTEXT variation (remove required data).
        
        Strategy:
        - Remove some/all items from Context
        - Update COT to reflect missing data (stateless language)
        - Change Decision to CALL (if it was ANSWER)
        - Update Tool Set to request missing data
        """
        missing = copy.deepcopy(example)
        
        original_context = missing.get("Context", [])
        original_decision = missing.get("Decision", "")
        
        if not original_context:
            # Context is already empty - create variant that needs to call tools
            missing["Context"] = []
            
            # Generate COT for empty context
            missing["COT"] = self._generate_missing_cot(
                query=missing.get("Q", ""),
                available_tools=missing.get("Tool Set", []),
                missing_all=True
            )
            
            # Ensure Decision is CALL
            if not original_decision.startswith("CALL"):
                missing["Decision"] = "CALL"
            
        else:
            # Remove 1-2 tool results (prefer removing from end)
            num_to_remove = min(len(original_context), random.choice([1, 2]))
            removed_tools = original_context[-num_to_remove:]
            missing["Context"] = original_context[:-num_to_remove]
            
            # Extract removed tool names
            removed_tool_names = [t["tool"] for t in removed_tools]
            
            # Update COT (stateless language)
            missing["COT"] = self._generate_missing_cot(
                query=missing.get("Q", ""),
                missing_tools=removed_tool_names,
                remaining_context=missing["Context"]
            )
            
            # Update Decision to CALL
            missing["Decision"] = "CALL"
            
            # Update Tool Set to request missing data
            if "Tool Set" not in missing:
                missing["Tool Set"] = []
            
            # Add removed tools to Tool Set if not already there
            for tool_name in removed_tool_names:
                if not any(t.get("name") == tool_name for t in missing["Tool Set"]):
                    missing["Tool Set"].append({
                        "name": tool_name,
                        "description": f"Retrieve {tool_name} data",
                        "parameters": {"type": "object", "properties": {}}
                    })
        
        # Update metadata
        if "metadata" not in missing:
            missing["metadata"] = {}
        
        removed_count = len(original_context) - len(missing.get("Context", []))
        removed_names = [t["tool"] for t in original_context[-removed_count:]] if removed_count > 0 else []
        
        missing["metadata"]["context_variation"] = "MISSING_CONTEXT"
        missing["metadata"]["variation_description"] = f"Removed {removed_count} tool result(s) from context"
        missing["metadata"]["removed_tools"] = removed_names
        missing["metadata"]["remaining_tools"] = [t["tool"] for t in missing.get("Context", [])]
        missing["metadata"]["expected_behavior"] = "Model identifies missing data and generates appropriate tool call"
        
        return missing
    
    def _create_extra_variation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create EXTRA_CONTEXT variation (add irrelevant data).
        
        Strategy:
        - Keep original Context
        - Add 1-2 irrelevant tool results
        - Update COT to mention filtering noise
        - Keep original Decision
        """
        extra = copy.deepcopy(example)
        
        original_context = extra.get("Context", [])
        
        # Get tools already in context
        existing_tools = {ctx["tool"] for ctx in original_context}
        
        # Filter irrelevant tools not already in context
        available_irrelevant = [
            t for t in self.IRRELEVANT_TOOLS 
            if t["tool"] not in existing_tools
        ]
        
        if not available_irrelevant:
            # No irrelevant tools to add, return complete variation
            logger.debug("No irrelevant tools available to add, skipping EXTRA_CONTEXT")
            return self._create_complete_variation(example)
        
        # Add 1-2 random irrelevant tools
        num_to_add = min(len(available_irrelevant), random.choice([1, 2]))
        extra_tools = random.sample(available_irrelevant, num_to_add)
        
        # Add to context
        extra["Context"] = original_context + extra_tools
        
        # Update COT to mention filtering
        original_cot = extra.get("COT", "")
        extra["COT"] = self._generate_extra_cot(
            original_cot=original_cot,
            extra_tools=[t["tool"] for t in extra_tools],
            relevant_tools=[ctx["tool"] for ctx in original_context]
        )
        
        # Update metadata
        if "metadata" not in extra:
            extra["metadata"] = {}
        
        extra["metadata"]["context_variation"] = "EXTRA_CONTEXT"
        extra["metadata"]["variation_description"] = f"Added {len(extra_tools)} irrelevant tool result(s)"
        extra["metadata"]["extra_tools"] = [t["tool"] for t in extra_tools]
        extra["metadata"]["relevant_tools"] = [ctx["tool"] for ctx in original_context]
        extra["metadata"]["expected_behavior"] = "Model filters noise and uses only relevant context"
        
        return extra
    
    def _can_create_missing_variation(self, example: Dict[str, Any]) -> bool:
        """
        Check if we can create a missing context variation.
        
        Can create if:
        - There's context to remove, OR
        - Decision is not ASK (can convert to CALL)
        """
        has_context = len(example.get("Context", [])) > 0
        decision = example.get("Decision", "")
        is_not_ask = not decision.startswith("ASK")
        
        return has_context or is_not_ask
    
    def _can_create_extra_variation(self, example: Dict[str, Any]) -> bool:
        """
        Check if we can create an extra context variation.
        
        Can create if there's already some context.
        """
        return len(example.get("Context", [])) > 0
    
    def _generate_missing_cot(
        self,
        query: str,
        missing_tools: List[str] = None,
        remaining_context: List[Dict] = None,
        available_tools: List[Dict] = None,
        missing_all: bool = False
    ) -> str:
        """
        Generate COT for missing context scenario using STATELESS language.
        
        Key: Use "Context contains/lacks" not "I called/received"
        """
        query_short = query[:80] + "..." if len(query) > 80 else query
        
        if missing_all or not missing_tools:
            # No context at all
            if available_tools and len(available_tools) > 0:
                tool_name = available_tools[0].get("name", "required tool")
            else:
                tool_name = "appropriate tool"
            
            return (
                f"To answer the query '{query_short}', I need specific data. "
                f"The context is currently empty - no information is available yet. "
                f"I should call {tool_name} to retrieve the necessary data."
            )
        
        # Some context missing
        if remaining_context and len(remaining_context) > 0:
            # Partial context available
            available = ", ".join([ctx["tool"] for ctx in remaining_context])
            missing = ", ".join(missing_tools) if missing_tools else "required data"
            
            return (
                f"The context currently contains data from {available}. "
                f"However, to fully answer '{query_short}', I also need data from {missing} "
                f"which is not present in the context. "
                f"I should call {missing_tools[0]} to get the missing information."
            )
        else:
            # All context missing
            missing = ", ".join(missing_tools) if missing_tools else "required data"
            
            return (
                f"To answer '{query_short}', I need data from {missing}. "
                f"This information is not currently available in the context. "
                f"I should call {missing_tools[0] if missing_tools else 'the appropriate tool'} to retrieve it."
            )
    
    def _generate_extra_cot(
        self,
        original_cot: str,
        extra_tools: List[str],
        relevant_tools: List[str]
    ) -> str:
        """
        Generate COT for extra context scenario using STATELESS language.
        
        Key: Mention filtering irrelevant data from context.
        """
        extra_mention = ", ".join(extra_tools)
        
        if relevant_tools:
            relevant_mention = ", ".join(relevant_tools)
            filter_note = (
                f"\n\nNote: The context also contains data from {extra_mention}, "
                f"but for this query, I only need information from {relevant_mention}. "
                f"I will focus on the relevant context and filter out the noise."
            )
        else:
            filter_note = (
                f"\n\nNote: The context contains data from {extra_mention}, "
                f"but this information is not directly relevant to answering the user's query. "
                f"I will focus on what's needed and ignore the irrelevant data."
            )
        
        return original_cot + filter_note


# Example usage for testing
if __name__ == "__main__":
    # Test with sample training example
    sample_example = {
        "Q": "What is my portfolio allocation?",
        "COT": "The user is asking for portfolio allocation data.",
        "Tool Set": [
            {
                "name": "get_portfolio_allocation",
                "description": "Retrieve portfolio allocation",
                "parameters": {}
            }
        ],
        "Decision": "CALL",
        "Context": [
            {
                "tool": "get_portfolio_allocation",
                "data": {"status": "success", "data": "60% stocks, 40% bonds"},
                "iteration": 0
            }
        ],
        "metadata": {"iteration": 1, "decision_type": "CALL"}
    }
    
    transformer = ContextVariationTransformer()
    variations = transformer.transform(sample_example)
    
    print(f"Generated {len(variations)} variations:")
    for var_type, var_example in variations.items():
        print(f"\n{'='*80}")
        print(f"VARIATION: {var_type}")
        print(f"{'='*80}")
        print(f"Context length: {len(var_example.get('Context', []))}")
        print(f"COT: {var_example.get('COT', '')[:150]}...")
        print(f"Decision: {var_example.get('Decision', '')[:50]}...")
        print(f"Metadata: {var_example.get('metadata', {})}")