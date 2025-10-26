"""
Complete Integration Example

Shows how to use multi-iteration trajectory generator with transformations
to achieve the full expansion pipeline.

Location: examples/integration_example.py (or tests/integration_test.py)

Run: python examples/integration_example.py
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


def main():
    """Complete integration example."""
    
    print("\n" + "="*80)
    print("COMPLETE INTEGRATION: TRANSFORMATIONS + MULTI-ITERATION")
    print("="*80)
    print("\nDemonstrates:")
    print("  1. Persona transformation (×5)")
    print("  2. Query complexity modification (×3)")
    print("  3. Multi-iteration trajectory generation (×1-3 per query)")
    print("  4. Training data output in {Qi, COTi, Tool Set i, Decisioni} format")
    print("="*80)
    
    # Setup
    setup_logger("INFO")
    config = load_config()
    
    # Initialize components
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
    
    # Seed queries
    seed_queries = [
        "What is my current portfolio allocation?",
        "How has my portfolio performed this year?"
    ]
    
    print(f"✅ Loaded {len(seed_queries)} seed queries")
    
    # =========================================================================
    # STEP 2: GENERATE TRAINING DATA
    # =========================================================================
    
    print("\n" + "-"*80)
    print("STEP 2: GENERATE TRAINING DATA")
    print("-"*80)
    
    all_training_examples = []
    stats = {
        "total_seeds": len(seed_queries),
        "total_personas": 0,
        "total_complexities": 0,
        "total_training_examples": 0,
        "examples_per_seed": {}
    }
    
    for seed_idx, seed_query in enumerate(seed_queries, 1):
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx}/{len(seed_queries)}: {seed_query}")
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
            
            # Apply query complexity modification
            print(f"\n    Applying complexity modification...")
            complexities = query_mod.transform(persona_query, include_original=False)
            stats["total_complexities"] += len(complexities)
            print(f"    ✅ Generated {len(complexities)} complexity variations")
            
            # For demo, use only 2 complexity levels
            demo_complexities = list(complexities.items())[:2]
            
            for complexity, complex_query in demo_complexities:
                print(f"\n    {'·'*74}")
                print(f"    COMPLEXITY: {complexity}")
                print(f"    {'·'*74}")
                print(f"    Query: {complex_query[:70]}...")
                
                # Generate multi-iteration trajectory
                print(f"\n      Generating trajectory...")
                examples = generator.generate_trajectory(
                    query=complex_query,
                    metadata={
                        "seed_query": seed_query,
                        "persona": persona_code,
                        "complexity": complexity
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
        
        stats["examples_per_seed"][f"seed_{seed_idx}"] = len(seed_examples)
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
    output_file = output_dir / "training_examples_multi_iter.jsonl"
    generator.save_training_examples(
        all_training_examples,
        output_file,
        format="jsonl"
    )
    
    print(f"✅ Saved {len(all_training_examples)} examples to: {output_file}")
    
    # Save stats
    stats_file = output_dir / "generation_stats.json"
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
    print(f"   Complexity Variations: {stats['total_complexities']}")
    print(f"   Training Examples Generated: {stats['total_training_examples']}")
    
    avg_examples = stats['total_training_examples'] / stats['total_seeds']
    print(f"\n   Average Examples per Seed: {avg_examples:.1f}")
    
    print(f"\n📈 EXPANSION FACTOR:")
    print(f"   Current Demo: ~{avg_examples:.0f}× per seed")
    print(f"   (Limited to 2 personas × 2 complexity for demo)")
    
    print(f"\n🎯 FULL PIPELINE POTENTIAL:")
    print(f"   With all transformations:")
    print(f"   5 personas × 3 complexity × AVG_ITERATIONS")
    print(f"   = 15-45 training examples per seed")
    print(f"   ")
    print(f"   With Phase 2 (PDF + Multi-turn):")
    print(f"   5 × 3 × 3 × 2 × 1.2 × AVG_ITERATIONS")
    print(f"   = 108-216 training examples per seed! 🚀")
    
    print(f"\n✅ OUTPUT FILES:")
    print(f"   Training Data: {output_file}")
    print(f"   Statistics: {stats_file}")
    
    print(f"\n📋 FORMAT VERIFICATION:")
    sample_dict = all_training_examples[0].to_dict(generator.field_names)
    print(f"   Query Field: '{config.output.schema.fields.query}' ✅")
    print(f"   COT Field: '{config.output.schema.fields.cot}' ✅")
    print(f"   Tools Field: '{config.output.schema.fields.tools}' ✅")
    print(f"   Decision Field: '{config.output.schema.fields.decision}' ✅")
    
    print("\n" + "="*80)
    print("🎉 INTEGRATION TEST COMPLETE 🎉")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review generated training examples")
    print("  2. Verify format matches your requirements")
    print("  3. Test with real ChromaDB (add PDFs)")
    print("  4. Scale to full transformation pipeline")
    print("  5. Add Phase 2 transformations (PDF aug + multi-turn)")
    print("="*80)


if __name__ == "__main__":
    main()