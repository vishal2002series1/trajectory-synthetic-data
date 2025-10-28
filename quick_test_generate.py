"""
Quick Test - Generate Trajectories from Seed Queries

This script tests trajectory generation with proper seed file parsing.
"""

import sys
import json
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.bedrock_provider import BedrockProvider
from src.core.vector_store import VectorStore
from src.generators.trajectory_generator_v2 import TrajectoryGeneratorV2
from src.utils import load_config, setup_logger, read_json

def main():
    """Run quick generation test."""
    
    print("\n" + "="*80)
    print("QUICK TEST: Trajectory Generation from Seeds")
    print("="*80)
    
    # Setup
    setup_logger("INFO")
    config = load_config()
    
    # Load seed queries
    print("\n" + "-"*80)
    print("STEP 1: Loading Seed Queries")
    print("-"*80)
    
    seed_file = Path("data/seed/data_seed_seed_queries.json")
    
    if not seed_file.exists():
        print(f"❌ Seed file not found: {seed_file}")
        return
    
    # Read JSON
    data = read_json(seed_file)
    
    # Extract seed_queries array
    if isinstance(data, dict) and 'seed_queries' in data:
        seed_queries = data['seed_queries']
        print(f"✅ Loaded {len(seed_queries)} seed queries from JSON object")
    elif isinstance(data, list):
        seed_queries = data
        print(f"✅ Loaded {len(seed_queries)} seed queries from JSON array")
    else:
        print(f"❌ Unexpected seed file format")
        return
    
    # Limit to 5 for testing
    test_queries = seed_queries[:5]
    
    print(f"\n📋 Testing with {len(test_queries)} queries:")
    for i, item in enumerate(test_queries, 1):
        if isinstance(item, dict):
            query = item.get('query', str(item))
        else:
            query = str(item)
        print(f"   {i}. {query[:60]}...")
    
    # Initialize components
    print("\n" + "-"*80)
    print("STEP 2: Initializing Components")
    print("-"*80)
    
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        embedding_model_id=config.bedrock.embedding_model_id,
        region=config.bedrock.region
    )
    print("✅ BedrockProvider initialized")
    
    vector_store = VectorStore(config)
    print(f"✅ VectorStore initialized")
    
    generator = TrajectoryGeneratorV2(
        bedrock_provider=provider,
        vector_store=vector_store,
        config=config
    )
    print(f"✅ TrajectoryGeneratorV2 initialized")
    print(f"   - Tools loaded: {len(generator.tools)}")
    print(f"   - Using mock tools: {generator.use_mock_tools}")
    
    # Generate trajectories
    print("\n" + "-"*80)
    print(f"STEP 3: Generating Trajectories ({len(test_queries)} queries)")
    print("-"*80)
    
    results = []
    success_count = 0
    
    for i, item in enumerate(test_queries, 1):
        # Extract query string
        if isinstance(item, dict):
            query = item.get('query', '')
            query_id = item.get('id', f'query_{i}')
        else:
            query = str(item)
            query_id = f'query_{i}'
        
        if not query:
            print(f"\n{i}. ⚠️  Skipping empty query")
            continue
        
        print(f"\n{i}. Query: {query[:70]}...")
        print(f"   ID: {query_id}")
        
        try:
            trajectory = generator.generate_trajectory(
                query=query,
                n_results=3,
                abstract=True
            )
            
            # Convert to output format
            output = generator.trajectory_to_output_format(
                trajectory=trajectory,
                include_metadata=True,
                include_tool_results=False
            )
            
            results.append({
                'id': query_id,
                'success': True,
                'output': output
            })
            
            success_count += 1
            print(f"   ✅ Success")
            print(f"   - Tools used: {[tc.name for tc in trajectory.tool_calls]}")
            print(f"   - Decision length: {len(trajectory.decision)} chars")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'id': query_id,
                'success': False,
                'error': str(e)
            })
    
    # Save results
    print("\n" + "-"*80)
    print("STEP 4: Saving Results")
    print("-"*80)
    
    output_dir = Path("data/output/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "test_trajectories.jsonl"
    
    with open(output_file, 'w') as f:
        for result in results:
            if result['success']:
                f.write(json.dumps(result['output']) + '\n')
    
    successful_outputs = [r for r in results if r['success']]
    print(f"✅ Saved {len(successful_outputs)} trajectories to: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\n📊 Statistics:")
    print(f"   Total Queries: {len(test_queries)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(test_queries) - success_count}")
    print(f"   Success Rate: {(success_count/len(test_queries)*100):.1f}%")
    
    if success_count > 0:
        print(f"\n✅ HEALTH CHECK PASSED")
        print(f"   System is working! Output saved to: {output_file}")
        
        # Show sample
        print(f"\n📄 Sample Output (First Result):")
        print("-"*80)
        sample = successful_outputs[0]['output']
        print(json.dumps(sample, indent=2)[:500] + "...")
    else:
        print(f"\n❌ HEALTH CHECK FAILED")
        print(f"   All generation attempts failed")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()