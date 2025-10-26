"""
Query Modification Transformer

Implements transformation T_query from the mathematical framework:
T_query: Q → {Q⁻, Q, Q⁺}

Where:
- Q⁻ = Query with reduced information (creates clarification need)
- Q  = Original query (unchanged)
- Q⁺ = Query with added complexity (additional requirements)

Expansion Factor: 1 → 3 (300% increase)

Mathematical Definition:
Q⁻ = remove_info(Q)     # Less specific, needs clarification
Q  = Q                   # Original
Q⁺ = add_requirements(Q) # More complex, additional constraints
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..utils import get_logger

logger = get_logger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels."""
    REDUCED = "Q-"      # Simplified, missing information
    ORIGINAL = "Q"      # Unchanged
    ENHANCED = "Q+"     # Increased complexity


@dataclass
class QueryModification:
    """Represents a modified query with metadata."""
    original: str
    modified: str
    complexity: QueryComplexity
    modifications_applied: List[str]


class QueryModifier:
    """
    Modify queries to adjust information completeness and complexity.
    
    Creates three variants:
    - Q⁻: Remove details to create clarification need
    - Q:  Keep original (identity transformation)
    - Q⁺: Add requirements to increase complexity
    """
    
    def __init__(self, bedrock_provider: Any):
        """
        Initialize QueryModifier.
        
        Args:
            bedrock_provider: BedrockProvider instance for LLM calls
        """
        self.provider = bedrock_provider
        logger.info("Initialized QueryModifier for Q⁻, Q, Q⁺ transformations")
    
    def transform(
        self,
        query: str,
        include_original: bool = True
    ) -> Dict[str, str]:
        """
        Transform query into complexity variations.
        
        Args:
            query: Original query (Q)
            include_original: Whether to include Q in output
            
        Returns:
            Dictionary mapping complexity → modified_query
            Example: {"Q-": "...", "Q": "...", "Q+": "..."}
        """
        logger.info(f"Generating query complexity variations")
        logger.debug(f"Original query: {query}")
        
        variations = {}
        
        # Q⁻: Reduced information
        try:
            q_minus = self._remove_information(query)
            variations["Q-"] = q_minus
            logger.debug(f"Q⁻: {q_minus}")
        except Exception as e:
            logger.error(f"Failed to generate Q⁻: {e}")
            variations["Q-"] = query
        
        # Q: Original (if requested)
        if include_original:
            variations["Q"] = query
            logger.debug(f"Q: {query}")
        
        # Q⁺: Enhanced complexity
        try:
            q_plus = self._add_requirements(query)
            variations["Q+"] = q_plus
            logger.debug(f"Q⁺: {q_plus}")
        except Exception as e:
            logger.error(f"Failed to generate Q⁺: {e}")
            variations["Q+"] = query
        
        logger.info(f"✅ Generated {len(variations)} complexity variations")
        return variations
    
    def _remove_information(self, query: str) -> str:
        """
        Generate Q⁻: Remove specific details to create clarification need.
        
        Strategies:
        - Remove specific numbers/amounts
        - Make references more vague
        - Remove constraining details
        - Simplify complex requests
        """
        prompt = f"""You are an expert at simplifying questions by removing specific details.

ORIGINAL QUESTION:
{query}

TASK:
Create a SIMPLER version by removing specific details, making the question more general.
This should create a need for clarification or follow-up questions.

STRATEGIES:
1. Remove specific numbers or amounts → Make general
2. Remove time references → Make vague
3. Remove specific constraints → Make open-ended
4. Simplify compound requests → Keep core only
5. Remove technical details → Use simpler terms

EXAMPLES:

Original: "What's my portfolio allocation and how does it compare to the 60/40 benchmark?"
Simplified: "What's my portfolio allocation?"

Original: "Calculate my projected retirement savings if I contribute $500/month for 20 years"
Simplified: "How much should I save for retirement?"

Original: "What are the tax implications of selling my tech stocks this quarter?"
Simplified: "What about selling some stocks?"

RULES:
1. Keep the CORE question recognizable
2. Make it LESS specific (remove details)
3. Should require follow-up clarification
4. Do NOT change the general topic
5. Output ONLY the simplified question

SIMPLIFIED QUESTION:"""
        
        simplified = self.provider.generate_text(
            prompt=prompt,
            max_tokens=150,
            temperature=0.6
        )
        
        return simplified.strip()
    
    def _add_requirements(self, query: str) -> str:
        """
        Generate Q⁺: Add complexity and additional requirements.
        
        Strategies:
        - Add specific metrics or constraints
        - Request comparisons or benchmarks
        - Add time dimensions
        - Request detailed breakdowns
        - Add conditional requirements
        """
        prompt = f"""You are an expert at making questions more comprehensive and complex.

ORIGINAL QUESTION:
{query}

TASK:
Create a MORE COMPLEX version by adding relevant requirements, constraints, or additional information needs.
This should make the question more sophisticated and detailed.

STRATEGIES:
1. Add specific metrics or targets
2. Request comparisons or benchmarks
3. Add time frames or horizons
4. Request breakdowns or categorizations
5. Add conditional constraints
6. Include related considerations

EXAMPLES:

Original: "What's my portfolio allocation?"
Enhanced: "What's my portfolio allocation and how does it compare to the target allocation for my risk tolerance, broken down by asset class and sector?"

Original: "How much should I save for retirement?"
Enhanced: "Based on my current age, income, and expected retirement date, how much should I contribute monthly to achieve a 70% income replacement rate in retirement, accounting for inflation and assuming 6% average returns?"

Original: "Should I rebalance my portfolio?"
Enhanced: "Should I rebalance my portfolio given current market conditions, and if so, what's the optimal rebalancing strategy to minimize tax impact while maintaining my target asset allocation?"

RULES:
1. Keep the SAME core question
2. Add relevant complexity (more requirements)
3. Make it require MORE detailed analysis
4. Stay realistic and domain-appropriate
5. Output ONLY the enhanced question

ENHANCED QUESTION:"""
        
        enhanced = self.provider.generate_text(
            prompt=prompt,
            max_tokens=250,
            temperature=0.7
        )
        
        return enhanced.strip()
    
    def transform_batch(
        self,
        queries: List[str],
        include_original: bool = True
    ) -> List[Dict[str, str]]:
        """
        Transform multiple queries into complexity variations.
        
        Args:
            queries: List of original queries
            include_original: Whether to include Q in output
            
        Returns:
            List of dictionaries, one per query
        """
        logger.info(f"Batch modifying {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries, 1):
            try:
                variations = self.transform(query, include_original)
                results.append(variations)
                logger.info(f"Completed {i}/{len(queries)}")
                
            except Exception as e:
                logger.error(f"Failed batch item {i}: {e}")
                results.append({"Q": query})
        
        logger.info(f"✅ Batch complete: {len(results)} items processed")
        return results
    
    def get_expansion_factor(self, include_original: bool = True) -> int:
        """Get the expansion factor."""
        return 3 if include_original else 2
    
    def __repr__(self) -> str:
        """String representation."""
        return "QueryModifier(variations=3: Q⁻, Q, Q⁺)"
    
    def transform_iterative(self, query: str, max_iterations: int = 5) -> List[str]:
        """
        Iteratively reduce query complexity.
        Returns: List of simplified queries [Q-1, Q-2, ...]
        """
        q_minus_chain = []
        current_query = query
        
        for iteration in range(max_iterations):
            # Generate simplified version
            simplified = self._remove_information(current_query)
            
            # Check if we actually simplified
            if simplified.strip().lower() == current_query.strip().lower():
                break
            
            q_minus_chain.append(simplified)
            
            # Ask LLM if we can simplify further
            if not self._can_simplify_further(simplified):
                break
            
            current_query = simplified
        
        return q_minus_chain if q_minus_chain else [query]

    def _can_simplify_further(self, query: str) -> bool:
        """Ask LLM if query can be simplified further"""
        prompt = f"""Can this query be simplified further while preserving core intent?

    Query: {query}

    Criteria:
    - YES if query has unnecessary words or can be more concise
    - NO if already at minimal form (1-3 words) or would lose meaning

    Answer ONLY: YES or NO"""
        
        response = self.provider.generate_text(
            prompt=prompt,
            max_tokens=10,
            temperature=0.3
        )
        return "YES" in response.strip().upper()


if __name__ == "__main__":
    # Test the modifier
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
    
    modifier = QueryModifier(provider)
    
    print("\n" + "="*80)
    print("QUERY MODIFIER TEST")
    print("="*80)
    
    # Test queries
    test_queries = [
        "What is my current portfolio allocation?",
        "How much should I save for retirement?",
        "Should I rebalance my portfolio now?"
    ]
    
    for test_query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Original Query: {test_query}")
        print(f"{'─'*80}")
        
        # Transform
        variations = modifier.transform(test_query)
        
        for complexity, modified in variations.items():
            print(f"\n{complexity}: {modified}")
    
    print("\n" + "="*80)
    print(f"✅ Generated {modifier.get_expansion_factor()} variations per query")
    print("="*80)