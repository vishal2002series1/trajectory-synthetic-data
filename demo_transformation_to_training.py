"""
Demonstration: Transformations → Training Data Pipeline

Shows how transformed queries are converted into final training format:
{Qi, COTi, Tool Set i, Decisioni}

This bridges:
1. Transformation modules (persona, query mod, tool data)
2. Trajectory generator (existing)
3. Final training data output

Usage:
    python demo_transformation_to_training.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.bedrock_provider import BedrockProvider
from src.core.vector_store import VectorStore
from src.transformations import PersonaTransformer, QueryModifier, ToolDataTransformer
from src.generators.trajectory_generator_v2 import TrajectoryGeneratorV2
from src.utils import load_config, setup_logger

# Setup
setup_logger("INFO")
config = load_config()

print("\n" + "="*80)
print("DEMONSTRATION: TRANSFORMATIONS → TRAINING DATA")
print("="*80)
print("\nGoal: Show how transformed queries become {Qi, COTi, Tool Set i, Decisioni}")
print("="*80)

# ============================================================================
# STEP 1: Initialize Components
# ============================================================================

print("\n" + "="*80)
print("STEP 1: INITIALIZING COMPONENTS")
print("="*80)

# Initialize Bedrock
provider = BedrockProvider(
    model_id=config.bedrock.model_id,
    region=config.bedrock.region
)
print(f"✅ BedrockProvider initialized")

# Initialize VectorStore (for trajectory generation)
vector_store = VectorStore(
    persist_directory=config.chromadb.persist_directory,
    collection_name=config.chromadb.collection_name,
    bedrock_provider=provider
)
print(f"✅ VectorStore initialized (collection: {config.chromadb.collection_name})")

# Initialize Transformers
persona_tx = PersonaTransformer(provider)
query_mod = QueryModifier(provider)
print(f"✅ Transformers initialized")

# Initialize Trajectory Generator (THIS IS THE KEY!)
trajectory_gen = TrajectoryGeneratorV2(
    bedrock_provider=provider,
    vector_store=vector_store,
    config=config
)
print(f"✅ TrajectoryGeneratorV2 initialized")

# ============================================================================
# STEP 2: Start with a Seed Query
# ============================================================================

print("\n" + "="*80)
print("STEP 2: SEED QUERY")
print("="*80)

seed_query = "What is my current portfolio allocation?"
print(f"\nOriginal Seed Query: {seed_query}")

# ============================================================================
# STEP 3: Apply Transformation (Example: Persona P1)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: APPLY TRANSFORMATION")
print("="*80)

# Get persona variations
print("\n3a. Generating persona variations...")
personas = persona_tx.transform(seed_query)

# Pick one persona for demonstration (P1 - First-time Investor)
transformed_query = personas["P1"]
print(f"\nSelected Persona: P1 (First-time Investor)")
print(f"Transformed Query: {transformed_query}")

# ============================================================================
# STEP 4: Generate Trajectory (THE MAGIC HAPPENS HERE!)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: GENERATE TRAJECTORY")
print("="*80)

print("\n4a. Calling trajectory generator with transformed query...")
print("    This will:")
print("    - Determine relevant tools")
print("    - Execute tool calls (search knowledge base)")
print("    - Generate Chain of Thought (COT)")
print("    - Generate final Decision/Answer")
print("    - Package as trajectory\n")

trajectory = trajectory_gen.generate_trajectory(
    query=transformed_query,
    n_results=3,
    abstract=True
)

print(f"✅ Trajectory generated!")
print(f"\nTrajectory Components:")
print(f"  - Query (Qi): {trajectory.query[:80]}...")
print(f"  - Chain of Thought (COTi): {trajectory.chain_of_thought[:80]}...")
print(f"  - Tools Used: {[tc.name for tc in trajectory.tool_calls]}")
print(f"  - Decision (Decisioni): {trajectory.decision[:80]}...")

# ============================================================================
# STEP 5: Convert to Training Format
# ============================================================================

print("\n" + "="*80)
print("STEP 5: CONVERT TO TRAINING FORMAT")
print("="*80)

print("\n5a. Converting trajectory to configured output format...")
print(f"    Output schema: {config.output.schema.type}")
print(f"    Field names: Qi={config.output.schema.fields.query}, "
      f"COTi={config.output.schema.fields.cot}, "
      f"Tool Set i={config.output.schema.fields.tools}, "
      f"Decisioni={config.output.schema.fields.decision}")

training_example = trajectory_gen.trajectory_to_output_format(
    trajectory=trajectory,
    include_metadata=True,
    include_tool_results=False
)

print("\n✅ Training example created!")
print("\n" + "="*80)
print("FINAL TRAINING DATA FORMAT")
print("="*80)

# Pretty print the training example
print("\n" + json.dumps(training_example, indent=2))

# ============================================================================
# STEP 6: Full Pipeline Demonstration (Multiple Variations)
# ============================================================================

print("\n" + "="*80)
print("STEP 6: FULL PIPELINE (30× EXPANSION)")
print("="*80)

print("\nDemonstrating how 1 seed → 30 training examples...")
print("(Showing 3 examples for brevity)\n")

training_examples = []
example_count = 0

# Generate a few examples to demonstrate
demo_personas = ["P1", "P2"]  # Just 2 personas for demo
demo_complexities = ["Q-", "Q"]  # Just 2 complexity levels for demo

for persona_code, persona_query in [(k, v) for k, v in personas.items() if k in demo_personas]:
    # Get query modifications
    query_mods = query_mod.transform(persona_query, include_original=True)
    
    for complexity, modified_query in [(k, v) for k, v in query_mods.items() if k in demo_complexities]:
        example_count += 1
        
        print(f"\n{'─'*80}")
        print(f"EXAMPLE {example_count}: {persona_code} + {complexity}")
        print(f"{'─'*80}")
        print(f"Query: {modified_query[:100]}...")
        
        # Generate trajectory
        traj = trajectory_gen.generate_trajectory(
            query=modified_query,
            n_results=3,
            abstract=True
        )
        
        # Convert to training format
        training_ex = trajectory_gen.trajectory_to_output_format(
            trajectory=traj,
            include_metadata=True,
            include_tool_results=False
        )
        
        training_examples.append(training_ex)
        
        print(f"\nTraining Example:")
        print(f"  Qi: {training_ex['Qi'][:80]}...")
        print(f"  COTi: {training_ex['COTi'][:80]}...")
        print(f"  Tool Set i: {[t['name'] for t in training_ex['Tool Set i']]}")
        print(f"  Decisioni: {training_ex['Decisioni'][:80]}...")
        
        if example_count >= 3:
            break
    
    if example_count >= 3:
        break

# ============================================================================
# STEP 7: Save Training Examples
# ============================================================================

print("\n" + "="*80)
print("STEP 7: SAVE TRAINING EXAMPLES")
print("="*80)

output_file = Path("data/output/demo_training_examples.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')

print(f"\n✅ Saved {len(training_examples)} training examples to: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: TRANSFORMATION → TRAINING DATA PIPELINE")
print("="*80)

print(f"""
PIPELINE STEPS:
1. Seed Query: "{seed_query}"
   ↓
2. Apply Transformations:
   - PersonaTransformer → 5 variations
   - QueryModifier → 3 variations each
   - ToolDataTransformer → 2 variations each
   ↓
3. For Each Transformed Query:
   - TrajectoryGeneratorV2.generate_trajectory()
     • Determines relevant tools
     • Executes tool calls (searches knowledge base)
     • Generates Chain of Thought
     • Generates Decision/Answer
   ↓
4. Convert to Training Format:
   - trajectory_to_output_format()
   - Output: {{Qi, COTi, Tool Set i, Decisioni}}
   ↓
5. Result: 30 training examples (5 × 3 × 2) per seed

EXPANSION MATH:
- 1 seed query
- × 5 personas (PersonaTransformer)
- × 3 complexity (QueryModifier: Q⁻, Q, Q⁺)
- × 2 tool data (ToolDataTransformer: correct, wrong)
= 30 training examples per seed ✅

TRAINING DATA FORMAT:
{{
  "Qi": "transformed query",
  "COTi": "chain of thought reasoning",
  "Tool Set i": [
    {{
      "name": "search_knowledge_base",
      "parameters": {{"query": "...", "n_results": 3}},
      "description": "Search for relevant info..."
    }}
  ],
  "Decisioni": "final answer based on tool results"
}}

OUTPUT LOCATION:
- {output_file}

NEXT STEPS:
1. Review the training examples above
2. Verify format matches your expectations
3. Confirm this is what you need for fine-tuning
4. Ready to scale to 90× with Phase 2!
""")

print("="*80)
print("🎉 DEMONSTRATION COMPLETE 🎉")
print("="*80)
print("\nYou can now:")
print("1. Review the generated training examples")
print("2. Inspect the JSONL file")
print("3. Confirm this format works for your fine-tuning")
print("4. Proceed to Phase 2 for 90× expansion!")
print("\n")