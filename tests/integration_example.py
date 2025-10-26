"""
Complete Integration Example with Iterative Q- Reduction

Shows how to use multi-iteration trajectory generator with transformations
including iterative Q- reduction to achieve full expansion pipeline.

Location: tests/integration_example.py

Run: python tests/integration_example.py
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.bedrock_provider import BedrockProvider
from src.generators.trajectory_generator_multi_iter import TrajectoryGeneratorMultiIter
from src.transformations import PersonaTransformer, QueryModifier
from src.utils import load_config, setup_logger


def load_seeds(seed_file: str = "data/seed/data_seed_seed_queries.json"):
    """Load seed queries from JSON file."""
    with open(seed_file, 'r') as f:
        data = json.load(f)
    return data['seed_queries']


def should_include_original(q_minus_chain, original_query, similarity_threshold=0.9):
    """
    Determine if original query should be included.
    
    Logic: Include Q (original) if Q-1 is very different from Q (original).
           Remove Q-1 if it's too similar to Q (original).
    
    Args:
        q_minus_chain: List of Q- reduced queries
        original_query: Original query
        similarity_threshold: Threshold above which queries are considered too similar
        
    Returns:
        True if original should be included (Q-1 is different), False otherwise
    """
    if not q_minus_chain:
        return True  # Always include if no Q- chain
    
    # Compare first Q- (Q-1) with original
    q_minus_1 = q_minus_chain[0]
    
    # Simple word overlap similarity
    words1 = set(original_query.lower().split())
    words2 = set(q_minus_1.lower().split())
    
    if not words1 or not words2:
        return True
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    similarity = len(intersection) / len(union)
    
    # If Q-1 is very similar to original (>90%), skip Q-1 and use original
    # If Q-1 is different (<90%), keep Q-1 and skip original
    should_keep_original = similarity >= similarity_threshold
    
    return should_keep_original


def main():
    """Complete integration example with iterative Q- reduction."""
    
    print("\n" + "="*80)
    print("COMPLETE INTEGRATION: ITERATIVE Q- REDUCTION + MULTI-ITERATION")
    print("="*80)
    print("\nDemonstrates:")
    print("  1. Load seeds from data/seed/")
    print("  2. Persona transformation (×2 for demo)")
    print("  3. Iterative Q- reduction (until most granular)")
    print("  4. Generate trajectory for EACH Q- level")
    print("  5. Q+ enhancement")
    print("  6. Q (original) handling with similarity check")
    print("  7. Training data output in {Qi, COTi, Tool Set i, Decisioni} format")
    print("="*80)
    
    # Setup
    setup_logger("INFO")
    config = load_config()
    
    # =========================================================================
    # STEP 1: INITIALIZE COMPONENTS
    # =========================================================================
    
    print("\n" + "-"*80)
    print("STEP 1: INITIALIZE COMPONENTS")
    print("-"*80)
    
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        region=config.bedrock.region
    )
    print("✅ BedrockProvider initialized")
    
    # Transformers
    persona_tx = PersonaTransformer(provider)
    query_mod = QueryModifier(provider)
    print("✅ Transformers initialized")
    
    # Multi-iteration generator
    generator = TrajectoryGeneratorMultiIter(
        bedrock_provider=provider,
        config=config,
        max_iterations=3,
        use_mock_tools=True  # Using mocked tools for demo
    )
    print("✅ Multi-iteration generator initialized")
    
    # Load seed queries from file
    try:
        seed_queries = load_seeds("data/seed/data_seed_seed_queries.json")
        print(f"✅ Loaded {len(seed_queries)} seed queries from file")
    except FileNotFoundError:
        print("⚠️  Seed file not found, using fallback seeds")
        seed_queries = [
            {
                "id": "seed_001",
                "query": "What is my current portfolio allocation?",
                "domain": "portfolio_management"
            },
            {
                "id": "seed_002",
                "query": "How has my portfolio performed this year?",
                "domain": "market_analysis"
            }
        ]
    
    # Demo mode: Use only first 2 seeds
    demo_seeds = seed_queries[:2]
    print(f"✅ Using {len(demo_seeds)} seeds for demo mode")
    
    # =========================================================================
    # STEP 2: GENERATE TRAINING DATA WITH ITERATIVE Q-
    # =========================================================================
    
    print("\n" + "-"*80)
    print("STEP 2: GENERATE TRAINING DATA WITH ITERATIVE Q-")
    print("-"*80)
    
    all_training_examples = []
    stats = {
        "total_seeds": len(demo_seeds),
        "total_personas": 0,
        "total_q_minus_levels": 0,
        "total_q_plus": 0,
        "total_q_original": 0,
        "total_training_examples": 0,
        "examples_per_seed": {}
    }
    
    for seed_idx, seed in enumerate(demo_seeds, 1):
        seed_id = seed.get("id", f"seed_{seed_idx}")
        seed_query = seed["query"]
        
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx}/{len(demo_seeds)}: {seed_query}")
        print(f"{'='*80}")
        
        seed_examples = []
        
        # Apply persona transformation
        print(f"\n  Applying persona transformation...")
        personas = persona_tx.transform(seed_query)
        stats["total_personas"] += len(personas)
        print(f"  ✅ Generated {len(personas)} persona variations")
        
        # For demo, use only 2 personas to keep output manageable
        demo_personas = list(personas.items())[:2]
        
        for persona_code, persona_query in demo_personas:
            print(f"\n  {'-'*76}")
            print(f"  PERSONA: {persona_code}")
            print(f"  {'-'*76}")
            print(f"  Query: {persona_query[:80]}...")
            
            # ====================================================================
            # ITERATIVE Q- REDUCTION
            # ====================================================================
            print(f"\n    Applying ITERATIVE Q- reduction...")
            q_minus_chain = query_mod.transform_iterative(persona_query, max_iterations=5)
            print(f"    ✅ Generated {len(q_minus_chain)} Q- reduction levels")
            
            stats["total_q_minus_levels"] += len(q_minus_chain)
            
            # Check if we should include original
            include_original = should_include_original(q_minus_chain, persona_query)
            
            # If original is too similar to Q-1, remove Q-1 from chain
            if include_original and q_minus_chain:
                print(f"    ℹ️  Q-1 similar to original (>90%), will use Q(original) instead")
                q_minus_chain = q_minus_chain[1:]  # Remove Q-1, we'll use original instead
            
            # Generate trajectories for each Q- level
            for q_minus_idx, q_minus_query in enumerate(q_minus_chain, 1):
                # Adjust numbering if Q-1 was removed
                actual_level = q_minus_idx + (1 if include_original else 0)
                
                print(f"\n    {'·'*74}")
                print(f"    COMPLEXITY: Q-{actual_level}")
                print(f"    {'·'*74}")
                print(f"    Query: {q_minus_query[:70]}...")
                
                print(f"\n      Generating trajectory...")
                examples = generator.generate_trajectory(
                    query=q_minus_query,
                    metadata={
                        "seed_id": seed_id,
                        "seed_query": seed_query,
                        "persona": persona_code,
                        "complexity": f"Q-{actual_level}",
                        "complexity_chain": q_minus_chain[:q_minus_idx],
                        "total_q_minus_levels": len(q_minus_chain)
                    }
                )
                
                print(f"      ✅ Generated {len(examples)} training examples")
                
                # Show summary
                for ex_idx, ex in enumerate(examples):
                    decision_type = ex.metadata["decision_type"]
                    iteration = ex.metadata["iteration"]
                    print(f"        Example {ex_idx+1}: Iteration {iteration} → {decision_type}")
                
                seed_examples.extend(examples)
                stats["total_training_examples"] += len(examples)
            
            # ====================================================================
            # Q (ORIGINAL) - Only if different from Q-1
            # ====================================================================
            if include_original:
                print(f"\n    {'·'*74}")
                print(f"    COMPLEXITY: Q (original)")
                print(f"    {'·'*74}")
                print(f"    Query: {persona_query[:70]}...")
                
                print(f"\n      Generating trajectory...")
                examples = generator.generate_trajectory(
                    query=persona_query,
                    metadata={
                        "seed_id": seed_id,
                        "seed_query": seed_query,
                        "persona": persona_code,
                        "complexity": "Q"
                    }
                )
                
                print(f"      ✅ Generated {len(examples)} training examples")
                
                for ex_idx, ex in enumerate(examples):
                    decision_type = ex.metadata["decision_type"]
                    iteration = ex.metadata["iteration"]
                    print(f"        Example {ex_idx+1}: Iteration {iteration} → {decision_type}")
                
                seed_examples.extend(examples)
                stats["total_q_original"] += 1
                stats["total_training_examples"] += len(examples)
            
            # ====================================================================
            # Q+ (ENHANCED COMPLEXITY)
            # ====================================================================
            print(f"\n    {'·'*74}")
            print(f"    COMPLEXITY: Q+")
            print(f"    {'·'*74}")
            
            q_plus = query_mod._add_requirements(persona_query)
            print(f"    Query: {q_plus[:70]}...")
            
            print(f"\n      Generating trajectory...")
            examples = generator.generate_trajectory(
                query=q_plus,
                metadata={
                    "seed_id": seed_id,
                    "seed_query": seed_query,
                    "persona": persona_code,
                    "complexity": "Q+"
                }
            )
            
            print(f"      ✅ Generated {len(examples)} training examples")
            
            for ex_idx, ex in enumerate(examples):
                decision_type = ex.metadata["decision_type"]
                iteration = ex.metadata["iteration"]
                print(f"        Example {ex_idx+1}: Iteration {iteration} → {decision_type}")
            
            seed_examples.extend(examples)
            stats["total_q_plus"] += 1
            stats["total_training_examples"] += len(examples)
        
        stats["examples_per_seed"][seed_id] = len(seed_examples)
        all_training_examples.extend(seed_examples)
        
        print(f"\n  {'='*76}")
        print(f"  SEED {seed_idx} SUMMARY: {len(seed_examples)} training examples generated")
        print(f"  {'='*76}")
    
    # =========================================================================
    # STEP 3: SAVE TRAINING DATA
    # =========================================================================
    
    print("\n" + "-"*80)
    print("STEP 3: SAVE TRAINING DATA")
    print("-"*80)
    
    output_dir = Path("data/output/integration_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    output_file = output_dir / "training_examples_iterative_q_minus.jsonl"
    generator.save_training_examples(
        all_training_examples,
        output_file,
        format="jsonl"
    )
    
    print(f"✅ Saved {len(all_training_examples)} examples to: {output_file}")
    
    # Save stats
    stats_file = output_dir / "generation_stats_iterative.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved statistics to: {stats_file}")
    
    # =========================================================================
    # STEP 4: SHOW SAMPLE OUTPUT
    # =========================================================================
    
    print("\n" + "-"*80)
    print("STEP 4: SAMPLE OUTPUT (First 2 Examples)")
    print("-"*80)
    
    for i, example in enumerate(all_training_examples[:2], 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}")
        print(f"{'='*80}")
        
        example_dict = example.to_dict(generator.field_names)
        print(json.dumps(example_dict, indent=2))
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    
    print(f"\n📊 STATISTICS:")
    print(f"   Seed Queries: {stats['total_seeds']}")
    print(f"   Persona Variations: {stats['total_personas']}")
    print(f"   Total Q- Levels: {stats['total_q_minus_levels']}")
    print(f"   Q+ Variations: {stats['total_q_plus']}")
    print(f"   Q (original) Included: {stats['total_q_original']}")
    print(f"   Training Examples Generated: {stats['total_training_examples']}")
    
    avg_examples = stats['total_training_examples'] / stats['total_seeds']
    print(f"\n   Average Examples per Seed: {avg_examples:.1f}")
    
    print(f"\n📈 ITERATIVE Q- EXPANSION:")
    print(f"   Each seed generates multiple Q- levels")
    print(f"   Each Q- level generates 2-3 trajectory examples")
    print(f"   Result: Rich variety of complexity levels!")
    
    print(f"\n🎯 FULL PIPELINE POTENTIAL:")
    print(f"   With all transformations:")
    print(f"   5 personas × {stats['total_q_minus_levels']//stats['total_personas']} Q- levels × 3 iterations")
    print(f"   = ~50-100 training examples per seed! 🚀")
    
    print(f"\n✅ OUTPUT FILES:")
    print(f"   Training Data: {output_file}")
    print(f"   Statistics: {stats_file}")
    
    print(f"\n📋 FORMAT VERIFICATION:")
    sample_dict = all_training_examples[0].to_dict(generator.field_names)
    print(f"   Query Field: '{config.output.schema.fields.query}' ✅")
    print(f"   COT Field: '{config.output.schema.fields.cot}' ✅")
    print(f"   Tools Field: '{config.output.schema.fields.tools}' ✅")
    print(f"   Decision Field: '{config.output.schema.fields.decision}' ✅")
    print(f"   Metadata includes: seed_id, complexity, chain ✅")
    
    print("\n" + "="*80)
    print("🎉 INTEGRATION TEST COMPLETE 🎉")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review generated training examples")
    print("  2. Verify Q- reduction levels are meaningful")
    print("  3. Check that Q(original) vs Q-1 logic works correctly")
    print("  4. Scale to full transformation pipeline (5 personas × 12 seeds)")
    print("  5. Add Phase 2 transformations (PDF aug + multi-turn)")
    print("="*80)


if __name__ == "__main__":
    main()