"""
Test script for Context Variation Transformer

Tests the stateless context variation transformer with sample data.

Usage:
    python test_context_variation.py
"""

import json
from src.transformations import ContextVariationTransformer
from src.utils import get_logger

logger = get_logger(__name__)


def test_basic_variation():
    """Test basic context variation with a simple example."""
    print("\n" + "="*80)
    print("TEST 1: Basic Context Variation")
    print("="*80)
    
    sample = {
        "Q": "What is my portfolio allocation?",
        "COT": "The user is asking for their portfolio allocation data. I can see from the context that portfolio allocation data is available.",
        "Tool Set": [],
        "Decision": "ANSWER: Your portfolio is 60% stocks and 40% bonds.",
        "Context": [
            {
                "tool": "get_portfolio_allocation",
                "data": {"status": "success", "data": "60% stocks, 40% bonds"},
                "iteration": 0,
                "timestamp": "2025-01-01T00:00:00"
            }
        ],
        "metadata": {"iteration": 1, "decision_type": "ANSWER"}
    }
    
    transformer = ContextVariationTransformer()
    variations = transformer.transform(sample)
    
    print(f"\n✅ Generated {len(variations)} variations\n")
    
    for var_type, variant in variations.items():
        print(f"\n{'─'*80}")
        print(f"VARIATION: {var_type}")
        print(f"{'─'*80}")
        print(f"Q: {variant['Q']}")
        print(f"Context length: {len(variant.get('Context', []))}")
        print(f"COT: {variant['COT'][:200]}...")
        print(f"Decision: {variant['Decision'][:100]}...")
        print(f"Variation metadata: {variant['metadata'].get('context_variation')}")
        print(f"Expected behavior: {variant['metadata'].get('expected_behavior')}")
        
        if var_type == "MISSING_CONTEXT":
            print(f"Removed tools: {variant['metadata'].get('removed_tools')}")
        elif var_type == "EXTRA_CONTEXT":
            print(f"Extra tools: {variant['metadata'].get('extra_tools')}")
    
    return variations


def test_empty_context():
    """Test with empty context (CALL decision)."""
    print("\n" + "="*80)
    print("TEST 2: Empty Context (CALL)")
    print("="*80)
    
    sample = {
        "Q": "What is my portfolio allocation?",
        "COT": "The user needs portfolio data.",
        "Tool Set": [
            {
                "name": "get_portfolio_allocation",
                "description": "Get portfolio data",
                "parameters": {}
            }
        ],
        "Decision": "CALL",
        "Context": [],  # Empty context
        "metadata": {"iteration": 0, "decision_type": "CALL"}
    }
    
    transformer = ContextVariationTransformer()
    variations = transformer.transform(sample)
    
    print(f"\n✅ Generated {len(variations)} variations\n")
    
    for var_type, variant in variations.items():
        print(f"\n{'─'*80}")
        print(f"VARIATION: {var_type}")
        print(f"{'─'*80}")
        print(f"Context length: {len(variant.get('Context', []))}")
        print(f"COT: {variant['COT']}")
        print(f"Decision: {variant['Decision']}")
    
    return variations


def test_multiple_context():
    """Test with multiple items in context."""
    print("\n" + "="*80)
    print("TEST 3: Multiple Context Items")
    print("="*80)
    
    sample = {
        "Q": "Compare my portfolio to benchmarks",
        "COT": "The user wants to compare their portfolio to benchmarks. I have both portfolio data and benchmark data in context.",
        "Tool Set": [],
        "Decision": "ANSWER: Your portfolio shows 60% stocks vs 70% benchmark...",
        "Context": [
            {
                "tool": "get_portfolio_allocation",
                "data": {"stocks": 60, "bonds": 40},
                "iteration": 0
            },
            {
                "tool": "search_knowledge_base",
                "data": {"benchmark": "70% stocks, 30% bonds"},
                "iteration": 0
            }
        ],
        "metadata": {"iteration": 1, "decision_type": "ANSWER"}
    }
    
    transformer = ContextVariationTransformer()
    variations = transformer.transform(sample)
    
    print(f"\n✅ Generated {len(variations)} variations\n")
    
    for var_type, variant in variations.items():
        print(f"\n{'─'*80}")
        print(f"VARIATION: {var_type}")
        print(f"{'─'*80}")
        print(f"Context length: {len(variant.get('Context', []))}")
        
        if variant.get('Context'):
            print(f"Context tools: {[c['tool'] for c in variant['Context']]}")
        
        print(f"COT (first 200 chars): {variant['COT'][:200]}...")
        
        if var_type == "MISSING_CONTEXT":
            print(f"Removed: {variant['metadata'].get('removed_tools')}")
            print(f"Remaining: {variant['metadata'].get('remaining_tools')}")
            print(f"Tool Set to call: {[t['name'] for t in variant.get('Tool Set', [])]}")
        elif var_type == "EXTRA_CONTEXT":
            print(f"Extra tools added: {variant['metadata'].get('extra_tools')}")
    
    return variations


def test_real_example():
    """Test with real example from your training data."""
    print("\n" + "="*80)
    print("TEST 4: Real Training Data Example")
    print("="*80)
    
    sample = {
        "Q": "How do I see what I own?",
        "COT": "The user wants to see what they own in their portfolio, but the current data only shows \"Mock data for get_portfolio_allocation\" which doesn't provide actual holdings information. I need to get the portfolio performance data which should include specific holdings and positions.",
        "Tool Set": [
            {
                "name": "get_portfolio_performance",
                "description": "Retrieve the user's portfolio performance metrics including returns, gains/losses, and comparisons to benchmarks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "period": {"type": "string", "description": "Time period for performance"}
                    }
                }
            }
        ],
        "Decision": "CALL",
        "Context": [
            {
                "tool": "get_portfolio_allocation",
                "parameters": {},
                "data": {"status": "success", "data": "Mock data for get_portfolio_allocation"},
                "iteration": 0,
                "timestamp": "2025-10-28T18:56:10.548610"
            }
        ],
        "metadata": {
            "iteration": 1,
            "decision_type": "CALL",
            "seed_query": "What is my current portfolio allocation?",
            "persona": "P1",
            "complexity": "Q-"
        }
    }
    
    transformer = ContextVariationTransformer()
    variations = transformer.transform(sample)
    
    print(f"\n✅ Generated {len(variations)} variations\n")
    
    # Save variations to file for inspection
    output_file = "test_variations_output.json"
    with open(output_file, 'w') as f:
        json.dump(variations, f, indent=2)
    
    print(f"💾 Saved variations to {output_file}")
    
    for var_type, variant in variations.items():
        print(f"\n{'─'*80}")
        print(f"VARIATION: {var_type}")
        print(f"{'─'*80}")
        print(f"Context length: {len(variant.get('Context', []))}")
        print(f"COT length: {len(variant['COT'])} chars")
        print(f"Decision: {variant['Decision'][:50]}...")
        print(f"Metadata variation: {variant['metadata'].get('context_variation')}")
    
    return variations


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CONTEXT VARIATION TRANSFORMER TESTS")
    print("="*80)
    
    try:
        # Test 1: Basic variation
        test_basic_variation()
        
        # Test 2: Empty context
        test_empty_context()
        
        # Test 3: Multiple context items
        test_multiple_context()
        
        # Test 4: Real example
        test_real_example()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNext steps:")
        print("1. Review test_variations_output.json to verify variations")
        print("2. Integrate into pipeline_commands.py")
        print("3. Run small pipeline test: python main.py pipeline --skip-ingest --limit 1")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()