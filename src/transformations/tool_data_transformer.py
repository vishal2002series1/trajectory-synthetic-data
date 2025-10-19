"""
Tool Data Variation Transformer

Implements transformation T_tool_data from the mathematical framework:
T_tool_data: (Q, Tools, A) → {(Q, Tools, D_correct, A), (Q, Tools, D_wrong, A_error)}

Where:
- D_correct = Correct tool data matching the user's context
- D_wrong = Mismatched tool data (wrong user, wrong parameters, etc.)
- A_error = Response indicating data mismatch detected

Purpose: Train the model to detect when tool responses don't match expectations.

Expansion Factor: 1 → 2 (200% increase)

Critical for production systems: LLM must detect when APIs return wrong data!
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import random

from ..utils import get_logger

logger = get_logger(__name__)


class DataMismatchType(Enum):
    """Types of data mismatches to generate."""
    WRONG_USER = "wrong_user"           # Data for different user
    WRONG_TIMEFRAME = "wrong_timeframe" # Data from wrong time period
    WRONG_ACCOUNT = "wrong_account"     # Data from wrong account
    PARTIAL_DATA = "partial_data"       # Incomplete data
    CONFLICTING = "conflicting"         # Internally inconsistent data


@dataclass
class ToolDataVariation:
    """Represents a tool data variation."""
    query: str
    tools_used: List[str]
    data_type: str  # "correct" or mismatch type
    tool_data: Dict[str, Any]
    expected_behavior: str  # What LLM should do
    decision: str  # Expected answer or error message


class ToolDataTransformer:
    """
    Generate variations with correct and mismatched tool data.
    
    Teaches the model to:
    1. Use correct data appropriately
    2. Detect data mismatches and errors
    3. Refuse to answer with wrong data
    4. Request correct data
    """
    
    def __init__(self, bedrock_provider: Any):
        """
        Initialize ToolDataTransformer.
        
        Args:
            bedrock_provider: BedrockProvider instance for LLM calls
        """
        self.provider = bedrock_provider
        logger.info("Initialized ToolDataTransformer for correct/wrong data variations")
    
    def transform(
        self,
        query: str,
        tools_used: List[str],
        correct_answer: str,
        user_context: Optional[Dict[str, Any]] = None,
        mismatch_types: Optional[List[DataMismatchType]] = None
    ) -> Dict[str, ToolDataVariation]:
        """
        Generate tool data variations (correct + wrong).
        
        Args:
            query: Original query
            tools_used: List of tools that were/will be used
            correct_answer: Expected correct answer
            user_context: User context for generating mismatches
            mismatch_types: Types of mismatches to generate
            
        Returns:
            Dictionary mapping variant_type → ToolDataVariation
            Example: {"correct": ..., "wrong_user": ..., ...}
        """
        logger.info(f"Generating tool data variations for query: {query[:50]}...")
        
        if user_context is None:
            user_context = self._extract_user_context(query)
        
        if mismatch_types is None:
            mismatch_types = [DataMismatchType.WRONG_USER]
        
        variations = {}
        
        # Generate CORRECT data variation
        correct_var = ToolDataVariation(
            query=query,
            tools_used=tools_used,
            data_type="correct",
            tool_data=self._generate_correct_data(query, tools_used, user_context),
            expected_behavior="Use the data to answer the question accurately",
            decision=correct_answer
        )
        variations["correct"] = correct_var
        logger.debug("Generated CORRECT data variation")
        
        # Generate WRONG data variations
        for mismatch_type in mismatch_types:
            wrong_var = self._generate_wrong_variation(
                query=query,
                tools_used=tools_used,
                user_context=user_context,
                mismatch_type=mismatch_type,
                correct_answer=correct_answer
            )
            variations[mismatch_type.value] = wrong_var
            logger.debug(f"Generated {mismatch_type.value} variation")
        
        logger.info(f"✅ Generated {len(variations)} tool data variations")
        return variations
    
    def _extract_user_context(self, query: str) -> Dict[str, Any]:
        """
        Extract user context from query.
        
        Identifies: user ID, possessive pronouns, temporal references.
        """
        context = {
            "has_possessive": any(word in query.lower() for word in ["my", "mine", "i"]),
            "has_temporal": any(word in query.lower() for word in ["current", "now", "today", "this"]),
            "user_id": "user_123",  # Default
            "account_id": "account_456",  # Default
        }
        
        return context
    
    def _generate_correct_data(
        self,
        query: str,
        tools_used: List[str],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate correct tool data that matches user context.
        """
        # Simulate correct tool responses
        tool_data = {}
        
        for tool_name in tools_used:
            if "portfolio" in tool_name.lower() or "allocation" in tool_name.lower():
                tool_data[tool_name] = {
                    "user_id": user_context.get("user_id", "user_123"),
                    "timestamp": "2025-10-20T10:00:00Z",
                    "data": {
                        "total_value": 487650,
                        "allocation": {
                            "stocks": 62,
                            "bonds": 30,
                            "cash": 8
                        }
                    },
                    "status": "success"
                }
            
            elif "risk" in tool_name.lower():
                tool_data[tool_name] = {
                    "user_id": user_context.get("user_id", "user_123"),
                    "risk_profile": "moderate",
                    "risk_score": 6.2,
                    "status": "success"
                }
            
            else:
                # Generic tool response
                tool_data[tool_name] = {
                    "user_id": user_context.get("user_id", "user_123"),
                    "status": "success",
                    "data": "Relevant data for the query"
                }
        
        return tool_data
    
    def _generate_wrong_variation(
        self,
        query: str,
        tools_used: List[str],
        user_context: Dict[str, Any],
        mismatch_type: DataMismatchType,
        correct_answer: str
    ) -> ToolDataVariation:
        """
        Generate a variation with mismatched/wrong tool data.
        """
        # Generate wrong tool data based on mismatch type
        wrong_data = self._generate_wrong_data(
            tools_used=tools_used,
            mismatch_type=mismatch_type,
            user_context=user_context
        )
        
        # Generate expected error detection response
        error_message = self._generate_error_detection_message(
            query=query,
            mismatch_type=mismatch_type,
            wrong_data=wrong_data
        )
        
        return ToolDataVariation(
            query=query,
            tools_used=tools_used,
            data_type=mismatch_type.value,
            tool_data=wrong_data,
            expected_behavior=f"Detect {mismatch_type.value} and refuse to answer",
            decision=error_message
        )
    
    def _generate_wrong_data(
        self,
        tools_used: List[str],
        mismatch_type: DataMismatchType,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate mismatched tool data based on type.
        """
        wrong_data = {}
        
        for tool_name in tools_used:
            if mismatch_type == DataMismatchType.WRONG_USER:
                # Data for different user
                wrong_data[tool_name] = {
                    "user_id": "user_999",  # Different user!
                    "timestamp": "2025-10-20T10:00:00Z",
                    "data": {
                        "total_value": 892000,  # Different values
                        "allocation": {
                            "stocks": 45,
                            "bonds": 40,
                            "cash": 15
                        }
                    },
                    "status": "success"
                }
            
            elif mismatch_type == DataMismatchType.WRONG_TIMEFRAME:
                # Data from wrong time period
                wrong_data[tool_name] = {
                    "user_id": user_context.get("user_id", "user_123"),
                    "timestamp": "2020-01-15T10:00:00Z",  # Old data!
                    "data": {
                        "total_value": 245000,
                        "allocation": {"stocks": 70, "bonds": 20, "cash": 10}
                    },
                    "status": "success"
                }
            
            elif mismatch_type == DataMismatchType.PARTIAL_DATA:
                # Incomplete data
                wrong_data[tool_name] = {
                    "user_id": user_context.get("user_id", "user_123"),
                    "timestamp": "2025-10-20T10:00:00Z",
                    "data": {
                        "total_value": 487650
                        # Missing allocation data!
                    },
                    "status": "partial_success",
                    "error": "Allocation data unavailable"
                }
            
            else:
                # Generic mismatch
                wrong_data[tool_name] = {
                    "user_id": "unknown",
                    "status": "error",
                    "error": f"Data mismatch: {mismatch_type.value}"
                }
        
        return wrong_data
    
    def _generate_error_detection_message(
        self,
        query: str,
        mismatch_type: DataMismatchType,
        wrong_data: Dict[str, Any]
    ) -> str:
        """
        Generate appropriate error detection message.
        
        The LLM should detect the mismatch and respond appropriately.
        """
        prompt = f"""You are an AI assistant that MUST verify data consistency before answering.

USER QUERY:
{query}

TOOL DATA RECEIVED:
{json.dumps(wrong_data, indent=2)}

PROBLEM DETECTED:
The tool data has a {mismatch_type.value} issue.

TASK:
Generate a response that:
1. Politely REFUSES to answer using this data
2. Clearly states the SPECIFIC problem detected
3. Explains what correct data would need
4. Does NOT attempt to answer the original question

EXAMPLE ERROR RESPONSES:

If wrong user:
"I notice the data returned is for user_id '999', but you asked about YOUR portfolio. I cannot provide your information using another user's data. I need to retrieve data specifically for your user account."

If wrong timeframe:
"The portfolio data I received is from January 2020, which is quite old. To answer your question about your CURRENT portfolio, I need up-to-date information from today."

If partial data:
"I received your portfolio value, but the allocation breakdown is missing. I cannot provide a complete answer without the allocation data. Let me retrieve the complete information."

YOUR ERROR RESPONSE:"""
        
        error_message = self.provider.generate_text(
            prompt=prompt,
            max_tokens=200,
            temperature=0.6
        )
        
        return error_message.strip()
    
    def transform_batch(
        self,
        items: List[Dict[str, Any]],
        mismatch_types: Optional[List[DataMismatchType]] = None
    ) -> List[Dict[str, ToolDataVariation]]:
        """
        Transform multiple items into tool data variations.
        
        Args:
            items: List of dicts with keys: query, tools_used, correct_answer
            mismatch_types: Types of mismatches to generate
            
        Returns:
            List of variation dictionaries
        """
        logger.info(f"Batch transforming {len(items)} items")
        
        results = []
        for i, item in enumerate(items, 1):
            try:
                variations = self.transform(
                    query=item["query"],
                    tools_used=item.get("tools_used", []),
                    correct_answer=item.get("correct_answer", ""),
                    user_context=item.get("user_context"),
                    mismatch_types=mismatch_types
                )
                results.append(variations)
                logger.info(f"Completed {i}/{len(items)}")
                
            except Exception as e:
                logger.error(f"Failed batch item {i}: {e}")
                results.append({})
        
        logger.info(f"✅ Batch complete: {len(results)} items processed")
        return results
    
    def get_expansion_factor(self, num_mismatch_types: int = 1) -> int:
        """Get the expansion factor."""
        return 1 + num_mismatch_types  # 1 correct + N wrong variations
    
    def __repr__(self) -> str:
        """String representation."""
        return "ToolDataTransformer(variations=2: correct + wrong)"


if __name__ == "__main__":
    # Test the transformer
    import sys
    sys.path.append('/home/claude')
    
    from src.core.bedrock_provider import BedrockProvider
    from src.utils import setup_logger, load_config
    
    setup_logger("INFO")
    config = load_config()
    
    # Initialize
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        region=config.bedrock.region
    )
    
    transformer = ToolDataTransformer(provider)
    
    print("\n" + "="*80)
    print("TOOL DATA TRANSFORMER TEST")
    print("="*80)
    
    # Test case
    test_query = "What is MY current portfolio value and allocation?"
    test_tools = ["get_portfolio_value", "get_allocation"]
    test_answer = "Your portfolio is worth $487,650 with 62% stocks, 30% bonds, 8% cash"
    
    print(f"\nQuery: {test_query}")
    print(f"Tools: {test_tools}")
    print(f"Correct Answer: {test_answer}\n")
    
    # Transform
    variations = transformer.transform(
        query=test_query,
        tools_used=test_tools,
        correct_answer=test_answer,
        mismatch_types=[DataMismatchType.WRONG_USER]
    )
    
    print("\nVARIATIONS GENERATED:")
    print("="*80)
    
    for var_type, variation in variations.items():
        print(f"\n{var_type.upper()}:")
        print(f"  Data Type: {variation.data_type}")
        print(f"  Expected Behavior: {variation.expected_behavior}")
        print(f"  Tool Data: {json.dumps(variation.tool_data, indent=4)}")
        print(f"  Decision: {variation.decision[:100]}...")
    
    print("\n" + "="*80)
    print(f"✅ Generated {len(variations)} variations")
    print("="*80)