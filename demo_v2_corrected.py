"""
Demo V2: Corrected Pipeline with Tool Definitions & Custom Output Format

Demonstrates:
1. ✅ Uses tool definitions from tools.json
2. ✅ Uses full config file settings
3. ✅ Outputs in format: {Qi, COTi, Tool Set i, Decisioni}
4. ✅ Abstraction layer still works
"""

import sys
import json
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.core import BedrockProvider, VectorStore
from src.generators import TrajectoryGeneratorV2, QuestionGenerator
from src.utils import load_config, setup_logger

print("\n" + "="*70)
print("DEMO V2: CORRECTED PIPELINE")
print("Tool Definitions + Custom Output Format")
print("="*70)

# Load config
config = load_config()
setup_logger(config.logging.level, config.logging.file)

print("\n📋 Configuration:")
print(f"   Tool definitions: {config.tools.definitions_file}")
print(f"   Output format: {config.output.schema.type}")
print(f"   Field names:")
print(f"     - Query: '{config.output.schema.fields.query}'")
print(f"     - COT: '{config.output.schema.fields.cot}'")
print(f"     - Tools: '{config.output.schema.fields.tools}'")
print(f"     - Decision: '{config.output.schema.fields.decision}'")

# Initialize components
print("\n" + "="*70)
print("STEP 1: Initialize Components")
print("="*70)

provider = BedrockProvider(
    model_id=config.bedrock.model_id,
    embedding_model_id=config.bedrock.embedding_model_id,
    region=config.bedrock.region
)

vector_store = VectorStore(config)

# Use V2 with tool definitions
trajectory_gen = TrajectoryGeneratorV2(
    bedrock_provider=provider,
    vector_store=vector_store,
    config=config
)

question_gen = QuestionGenerator(bedrock_provider=provider)

print("✅ All components initialized")
print(f"✅ Loaded {len(trajectory_gen.tools)} tool definitions")

# Show tool definitions
print("\n" + "="*70)
print("TOOL DEFINITIONS LOADED")
print("="*70)

for i, tool in enumerate(trajectory_gen.tools, 1):
    print(f"\n{i}. {tool['name']}")
    print(f"   Description: {tool['description']}")
    print(f"   Parameters: {', '.join(tool['parameters']['properties'].keys())}")

# Test query
print("\n" + "="*70)
print("STEP 2: Generate Test Trajectory")
print("="*70)

test_query = "How does time horizon affect the importance of investment returns versus savings?"

print(f"\n📝 Test Query:")
print(f"   {test_query}")

print(f"\n⏳ Generating trajectory...")

trajectory = trajectory_gen.generate_trajectory(
    query=test_query,
    n_results=3,
    abstract=True
)

print(f"\n✅ Trajectory generated!")

# Display trajectory
print("\n" + "="*70)
print("TRAJECTORY DETAILS")
print("="*70)

print(f"\n📝 Qi (Query):")
print(f"   {trajectory.query}")

print(f"\n💭 COTi (Chain of Thought):")
print(f"   {trajectory.chain_of_thought}")

print(f"\n🔧 Tool Set i:")
for i, tc in enumerate(trajectory.tool_calls, 1):
    print(f"\n   Tool {i}:")
    print(f"     Name: {tc.name}")
    print(f"     Description: {tc.description}")
    print(f"     Parameters:")
    for key, value in tc.parameters.items():
        print(f"       - {key}: {value}")
    if tc.result:
        print(f"     Result: {tc.result}")

print(f"\n✅ Decisioni (Final Answer - ABSTRACTED):")
print(f"   {trajectory.decision}")

print(f"\n📊 Metadata:")
for key, value in trajectory.metadata.items():
    print(f"   - {key}: {value}")

# Show output format
print("\n" + "="*70)
print("OUTPUT FORMAT (as configured)")
print("="*70)

output = trajectory_gen.trajectory_to_output_format(trajectory)

print("\n" + json.dumps(output, indent=2))

# Verify format
print("\n" + "="*70)
print("FORMAT VERIFICATION")
print("="*70)

expected_fields = [
    config.output.schema.fields.query,
    config.output.schema.fields.cot,
    config.output.schema.fields.tools,
    config.output.schema.fields.decision
]

print(f"\n✅ Expected fields: {expected_fields}")
print(f"✅ Actual fields: {list(output.keys())}")

all_present = all(field in output for field in expected_fields)
print(f"\n{'✅' if all_present else '❌'} All required fields present: {all_present}")

# Check abstraction
print("\n" + "="*70)
print("ABSTRACTION VERIFICATION")
print("="*70)

import re
decision = output[config.output.schema.fields.decision]

has_figure = bool(re.search(r'\b[Ff]igure\s+\d+\b', decision))
has_page = bool(re.search(r'\b[Pp]age\s+\d+\b', decision))

if has_figure or has_page:
    print(f"\n⚠️  WARNING: Document references found!")
    if has_figure:
        print(f"   - Found figure reference")
    if has_page:
        print(f"   - Found page reference")
else:
    print(f"\n✅ No figure/page references found!")
    print(f"✅ Answer is generic and transferable")

# Generate multiple trajectories
print("\n" + "="*70)
print("STEP 3: Generate Multiple Trajectories")
print("="*70)

test_queries = [
    "What's the relationship between risk and return?",
    "How does diversification reduce risk?",
    "Why are costs important in investing?"
]

print(f"\n⏳ Generating {len(test_queries)} trajectories...")

trajectories = trajectory_gen.generate_batch(
    queries=test_queries,
    n_results=3,
    abstract=True
)

print(f"\n✅ Generated {len(trajectories)} trajectories")

# Save to file
print("\n" + "="*70)
print("STEP 4: Save to File")
print("="*70)

output_path = Path(config.output.output_dir) / "synthetic_trajectories_v2.jsonl"

trajectory_gen.save_trajectories(
    trajectories=trajectories,
    output_path=str(output_path)
)

print(f"\n✅ Saved {len(trajectories)} trajectories to:")
print(f"   {output_path}")

# Show sample from file
print(f"\n📄 Sample from output file:")
print("="*70)

with open(output_path, 'r') as f:
    first_line = f.readline()
    sample = json.loads(first_line)

print(json.dumps(sample, indent=2))

# Summary
print("\n" + "="*70)
print("✅ DEMO COMPLETE!")
print("="*70)

print(f"\nKey Improvements:")
print(f"  1. ✅ Uses tool definitions from {config.tools.definitions_file}")
print(f"  2. ✅ Uses config.output.schema for field names")
print(f"  3. ✅ Outputs in format: {{Qi, COTi, Tool Set i, Decisioni}}")
print(f"  4. ✅ Tool calls include full definitions")
print(f"  5. ✅ Abstraction still removes figure/page references")
print(f"  6. ✅ All config settings respected")

print(f"\nOutput Structure:")
print(f"  {config.output.schema.fields.query}: Query/question")
print(f"  {config.output.schema.fields.cot}: Chain of thought reasoning")
print(f"  {config.output.schema.fields.tools}: Tool calls with parameters")
print(f"  {config.output.schema.fields.decision}: Final answer (abstracted)")

print("\n" + "="*70 + "\n")