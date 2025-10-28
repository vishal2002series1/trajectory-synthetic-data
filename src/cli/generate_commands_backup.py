"""
Generate commands for synthetic trajectory generation.

Handles:
- Generation from seed queries
- Generation without seeds (from scratch)
- Output in configured format
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..core import BedrockProvider, VectorStore
from ..generators import TrajectoryGeneratorV2
from ..utils import get_logger, ensure_dir, write_jsonl, read_json, read_jsonl
# from ..utils import load_config, get_logger, read_json, read_jsonl, write_jsonl

logger = get_logger(__name__)


class GenerateCommand:
    """Command handler for trajectory generation."""
    
    def __init__(self, config):
        """
        Initialize generate command.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info("GenerateCommand initialized")
    
    def generate(
        self,
        seed_file: Optional[str] = None,
        no_seed: bool = False,
        output: Optional[str] = None,
        limit: Optional[int] = None,
        format: str = 'jsonl'
    ) -> int:
        """
        Generate synthetic trajectories.
        
        Args:
            seed_file: Path to seed queries file
            no_seed: Generate without seeds
            output: Output directory
            limit: Maximum number of examples
            format: Output format
            
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        if not seed_file and not no_seed:
            print("❌ Error: Must provide either seed_file or --no-seed flag")
            return 1
        
        print(f"\n{'='*80}")
        print("TRAJECTORY GENERATION")
        print(f"{'='*80}")
        
        if no_seed:
            print(f"\n⚙️  Mode: NO-SEED generation")
            print(f"   Will generate queries from ChromaDB content")
            print(f"   Target: {self.config.generation.target_qa_pairs} QA pairs")
        else:
            seed_path = Path(seed_file)
            if not seed_path.exists():
                print(f"❌ Error: Seed file not found: {seed_path}")
                return 1
            
            print(f"\n📄 Seed file: {seed_path.name}")
            print(f"   Path: {seed_path.absolute()}")
        
        if limit:
            print(f"   Limit: {limit} examples")
        
        # Prompt for confirmation
        response = input(f"\nProceed with generation? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Generation cancelled")
            return 1
        
        try:
            # Initialize components
            print(f"\n{'─'*80}")
            print("STEP 1: Initializing Components")
            print(f"{'─'*80}")
            
            provider = BedrockProvider(
                model_id=self.config.bedrock.model_id,
                embedding_model_id=self.config.bedrock.embedding_model_id,
                region=self.config.bedrock.region
            )
            print("✅ Bedrock provider initialized")
            
            vector_store = VectorStore(config=self.config)
            print(f"✅ VectorStore initialized")
            print(f"   Collection: {self.config.chromadb.collection_name}")
            print(f"   Documents: {vector_store.count()}")
            
            if vector_store.count() == 0:
                print("\n⚠️  Warning: ChromaDB is empty!")
                print("   You need to ingest PDFs first using:")
                print("   python main.py ingest <pdf_path>")
                return 1
            
            generator = TrajectoryGeneratorV2(
                bedrock_provider=provider,
                vector_store=vector_store,
                config=self.config,
                use_mock_tools=False
            )
            print("✅ Trajectory generator initialized")
            
            # Load or generate queries
            print(f"\n{'─'*80}")
            print("STEP 2: Loading Queries")
            print(f"{'─'*80}")
            
            if no_seed:
                # Generate queries from ChromaDB content
                print("⚙️  Generating queries from document content...")
                print("   (This feature is under development)")
                print("   For now, please use seed queries")
                return 1
            else:
                # Load seed queries - support both .json and .jsonl
                seed_path = Path(seed_file)
                
                if seed_path.suffix == '.jsonl':
                    seed_data = read_jsonl(seed_file)
                else:
                    seed_data = read_json(seed_file)
                
                # Extract queries
                if isinstance(seed_data, dict) and 'seeds' in seed_data:
                    queries = [item['query'] for item in seed_data['seeds']]
                elif isinstance(seed_data, list):
                    queries = [item['query'] if isinstance(item, dict) else item 
                              for item in seed_data]
                else:
                    print(f"❌ Error: Invalid seed file format")
                    return 1
                
                print(f"✅ Loaded {len(queries)} seed queries")
                
                if limit:
                    queries = queries[:limit]
                    print(f"   Limited to {len(queries)} queries")
            
            # Generate trajectories
            print(f"\n{'─'*80}")
            print("STEP 3: Generating Trajectories")
            print(f"{'─'*80}\n")
            
            trajectories = []
            
            for i, query in enumerate(queries, 1):
                print(f"[{i}/{len(queries)}] Generating trajectory for:")
                print(f"   {query[:80]}...")
                
                try:
                    trajectory = generator.generate_trajectory(
                        query=query,
                        n_results=3,
                        abstract=True
                    )
                    
                    # Convert to output format
                    output_example = generator.trajectory_to_output_format(
                        trajectory=trajectory,
                        include_metadata=self.config.output.schema.include_metadata,
                        include_tool_results=self.config.output.schema.include_tool_results
                    )
                    
                    trajectories.append(output_example)
                    print(f"   ✅ Generated\n")
                
                except Exception as e:
                    print(f"   ❌ Error: {e}\n")
                    logger.error(f"Failed to generate trajectory for query {i}: {e}")
            
            print(f"{'─'*80}")
            print(f"✅ GENERATION COMPLETE")
            print(f"{'─'*80}")
            print(f"\nGenerated {len(trajectories)} trajectories")
            
            # Save results
            if not output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = f"data/output/trajectories/{timestamp}"
            
            output_dir = Path(output)
            ensure_dir(output_dir)
            
            if format in ['json', 'both']:
                json_file = output_dir / "trajectories.json"
                with open(json_file, 'w') as f:
                    json.dump(trajectories, f, indent=2)
                print(f"💾 Saved JSON: {json_file}")
            
            if format in ['jsonl', 'both']:
                jsonl_file = output_dir / "trajectories.jsonl"
                write_jsonl(trajectories, str(jsonl_file))
                print(f"💾 Saved JSONL: {jsonl_file}")
            
            # Summary
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            print(f"\n📊 Generation Statistics:")
            print(f"   Queries processed: {len(queries)}")
            print(f"   Trajectories generated: {len(trajectories)}")
            print(f"   Success rate: {len(trajectories)/len(queries)*100:.1f}%")
            print(f"\n💾 Output saved to: {output_dir}")
            
            return 0
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Generation failed: {e}", exc_info=True)
            return 1
