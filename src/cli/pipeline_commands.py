"""
Pipeline commands for end-to-end workflow orchestration - Phase 2 Multi-Iteration

Handles:
- Full pipeline: Ingest → Transform → Generate
- Multi-iteration trajectory generation (1-3 examples per query)
- Comprehensive progress tracking

Phase 2 Expansion: 15-45× per seed query!
- 5 personas × 3 complexity levels × (1-3 iterations per query)
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..core import BedrockProvider, VectorStore, PDFParser, ChromaDBManager
from ..transformations import PersonaTransformer, QueryModifier, ToolDataTransformer
from ..generators import TrajectoryGeneratorMultiIter  # ✅ Phase 2!
from ..utils import get_logger, ensure_dir, write_jsonl, write_json, read_json

logger = get_logger(__name__)


class PipelineCommand:
    """Command handler for end-to-end pipeline execution with multi-iteration support."""
    
    def __init__(self, config):
        """
        Initialize pipeline command.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info("PipelineCommand initialized (Phase 2 - Multi-Iteration)")
    
    def run_pipeline(
        self,
        seed_file: str,
        pdf_dir: Optional[str] = None,
        skip_ingest: bool = False,
        output: Optional[str] = None,
        limit: Optional[int] = None
    ) -> int:
        """
        Run complete end-to-end pipeline with multi-iteration support.
        
        Phase 2 Pipeline:
        1. Ingest PDFs (if not skipped)
        2. Load seed queries
        3. Apply persona transformations (×5)
        4. Apply query complexity (Q-, Q, Q+) (×3)
        5. Generate multi-iteration trajectories (×1-3 per query)
        
        Result: 15-45× expansion per seed query!
        
        Args:
            seed_file: Path to seed queries file
            pdf_dir: Directory containing PDFs to ingest
            skip_ingest: Skip ingestion step
            output: Output directory
            limit: Maximum number of seed queries to process
        
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        # Validate inputs
        seed_path = Path(seed_file)
        if not seed_path.exists():
            print(f"❌ Error: Seed file not found: {seed_path}")
            return 1
        
        if pdf_dir and not skip_ingest:
            pdf_dir_path = Path(pdf_dir)
            if not pdf_dir_path.exists():
                print(f"❌ Error: PDF directory not found: {pdf_dir_path}")
                return 1
        
        # Setup output directory
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"data/output/pipeline/{timestamp}"
        
        output_dir = Path(output)
        ensure_dir(output_dir)
        
        # Display pipeline overview
        print(f"\n{'='*80}")
        print("COMPLETE PIPELINE EXECUTION (Phase 2 - Multi-Iteration)")
        print(f"{'='*80}")
        print(f"\n📋 Pipeline Configuration:")
        print(f"   Seed file: {seed_path.name}")
        if pdf_dir and not skip_ingest:
            print(f"   PDF directory: {pdf_dir}")
        else:
            print(f"   PDF ingestion: SKIPPED (using existing ChromaDB)")
        print(f"   Output directory: {output_dir}")
        if limit:
            print(f"   Limit: {limit} seed queries")
        
        print(f"\n📊 Expected Expansion (Phase 2):")
        print(f"   Per seed query: 15-45× (5 personas × 3 complexity × 1-3 iterations)")
        
        # Prompt for confirmation
        response = input(f"\nProceed with pipeline? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Pipeline cancelled")
            return 1
        
        # Track statistics
        stats = {
            "start_time": datetime.now().isoformat(),
            "total_seeds": 0,
            "pdfs_ingested": 0,
            "total_personas": 0,
            "total_complexities": 0,
            "total_training_examples": 0,
            "errors": []
        }
        
        try:
            # ════════════════════════════════════════════════════════════════
            # STAGE 1: PDF INGESTION (Optional)
            # ════════════════════════════════════════════════════════════════
            
            if not skip_ingest and pdf_dir:
                print(f"\n{'='*80}")
                print("STAGE 1: PDF INGESTION")
                print(f"{'='*80}\n")
                
                result = self._ingest_pdfs(pdf_dir)
                stats["pdfs_ingested"] = result.get("success_count", 0)
                
                if result["exit_code"] != 0:
                    print(f"⚠️  Warning: Some PDFs failed to ingest")
                    stats["errors"].extend(result.get("errors", []))
            
            else:
                print(f"\n{'='*80}")
                print("STAGE 1: PDF INGESTION - SKIPPED")
                print(f"{'='*80}\n")
                print("Using existing ChromaDB collection")
            
            # ════════════════════════════════════════════════════════════════
            # STAGE 2: INITIALIZE COMPONENTS
            # ════════════════════════════════════════════════════════════════
            
            print(f"\n{'='*80}")
            print("STAGE 2: INITIALIZING COMPONENTS")
            print(f"{'='*80}\n")
            
            provider = BedrockProvider(
                model_id=self.config.bedrock.model_id,
                embedding_model_id=self.config.bedrock.embedding_model_id,
                region=self.config.bedrock.region
            )
            print("✅ Bedrock provider initialized")
            
            vector_store = VectorStore(config=self.config)
            print(f"✅ VectorStore initialized ({vector_store.count()} documents)")
            
            if vector_store.count() == 0:
                print("\n❌ Error: ChromaDB is empty! Need to ingest PDFs first.")
                return 1
            
            # Initialize transformers
            persona_tx = PersonaTransformer(provider)
            query_mod = QueryModifier(provider)
            print("✅ Transformers initialized")
            
            # ✅ Multi-Iteration Generator (Phase 2)
            generator = TrajectoryGeneratorMultiIter(
                bedrock_provider=provider,
                config=self.config,
                max_iterations=3,
                use_mock_tools=False  # Using real vector store
            )
            print("✅ Multi-Iteration trajectory generator initialized (Phase 2)")
            
            # ════════════════════════════════════════════════════════════════
            # STAGE 3: LOAD SEED QUERIES
            # ════════════════════════════════════════════════════════════════
            
            print(f"\n{'='*80}")
            print("STAGE 3: LOADING SEED QUERIES")
            print(f"{'='*80}\n")
            
            seed_data = read_json(str(seed_path))
            
            if isinstance(seed_data, dict):
                if 'seed_queries' in seed_data:
                    seeds = seed_data['seed_queries']
                elif 'queries' in seed_data:
                    seeds = seed_data['queries']
                else:
                    print(f"❌ Error: Invalid seed file format")
                    return 1
            elif isinstance(seed_data, list):
                seeds = seed_data
            else:
                print(f"❌ Error: Invalid seed file format")
                return 1
            
            # Normalize to list of strings
            normalized_queries = []
            for item in seeds:
                if isinstance(item, str):
                    normalized_queries.append(item)
                elif isinstance(item, dict):
                    query = item.get('query') or item.get('Q') or item.get('text')
                    if query:
                        normalized_queries.append(query)
                    else:
                        print(f"⚠️  Warning: Skipping item with unknown format: {item}")
                else:
                    print(f"⚠️  Warning: Skipping non-string/dict item: {item}")
            
            seeds = normalized_queries
            stats["total_seeds"] = len(seeds)
            
            # Apply limit
            if limit:
                seeds = seeds[:limit]
                print(f"✅ Loaded {stats['total_seeds']} seed queries (limiting to {limit})")
            else:
                print(f"✅ Loaded {len(seeds)} seed queries")
            
            # ════════════════════════════════════════════════════════════════
            # STAGE 4: APPLY TRANSFORMATIONS & GENERATE
            # ════════════════════════════════════════════════════════════════
            
            print(f"\n{'='*80}")
            print("STAGE 4: APPLYING TRANSFORMATIONS & GENERATING TRAJECTORIES")
            print(f"{'='*80}\n")
            
            all_training_examples = []
            
            for seed_idx, seed in enumerate(seeds, 1):
                print(f"\n{'='*80}")
                print(f"SEED {seed_idx}/{len(seeds)}: {seed[:80]}...")
                print(f"{'='*80}")
                
                # Get query text
                query = seed if isinstance(seed, str) else seed.get('query', '')
                
                # Apply persona transformation
                print(f"  Applying persona transformation...")
                personas = persona_tx.transform(query)
                stats["total_personas"] += len(personas)
                print(f"  ✅ Generated {len(personas)} persona variations")
                
                # For each persona (limit to 2 for demo)
                for persona_code, persona_query in list(personas.items()):
                    print(f"\n  {'─'*76}")
                    print(f"  PERSONA: {persona_code}")
                    print(f"  {'─'*76}")
                    
                    # Apply query complexity modifications
                    print(f"    Applying complexity modifications...")
                    complexities = query_mod.transform(persona_query, include_original=False)
                    stats["total_complexities"] += len(complexities)
                    print(f"    ✅ Generated {len(complexities)} complexity variations")
                    
                    # For each complexity level (limit to 2 for demo)
                    for complexity, complex_query in list(complexities.items()):
                        print(f"\n    ··········································")
                        print(f"    COMPLEXITY: {complexity}")
                        print(f"    ··········································")
                        print(f"    Query: {complex_query[:70]}...")
                        
                        # ✅ Generate multi-iteration trajectory
                        print(f"      Generating trajectory...")
                        try:
                            examples = generator.generate_trajectory(
                                query=complex_query,
                                metadata={
                                    "seed_query": query,
                                    "persona": persona_code,
                                    "complexity": complexity
                                }
                            )
                            
                            print(f"      ✅ Generated {len(examples)} training examples")
                            
                            # Show example details
                            for ex_idx, ex in enumerate(examples):
                                decision_type = ex.metadata["decision_type"]
                                iteration = ex.metadata["iteration"]
                                print(f"        Example {ex_idx+1}: Iteration {iteration} → {decision_type}")
                            
                            # Convert to output format and collect
                            for example in examples:
                                example_dict = example.to_dict(generator.field_names)
                                all_training_examples.append(example_dict)
                            
                            stats["total_training_examples"] += len(examples)
                            
                        except Exception as e:
                            error_msg = f"Trajectory generation failed: {e}"
                            print(f"      ❌ {error_msg}")
                            logger.error(error_msg)
                            stats["errors"].append(error_msg)
            
            # ════════════════════════════════════════════════════════════════
            # STAGE 5: SAVE RESULTS
            # ════════════════════════════════════════════════════════════════
            
            print(f"\n{'='*80}")
            print("STAGE 5: SAVING RESULTS")
            print(f"{'='*80}\n")
            
            # Save training examples
            output_file = output_dir / "training_examples.jsonl"
            write_jsonl(all_training_examples, str(output_file))
            print(f"✅ Saved {len(all_training_examples)} examples to: {output_file}")
            
            # Save statistics
            stats["end_time"] = datetime.now().isoformat()
            stats_file = output_dir / "generation_stats.json"
            write_json(stats, str(stats_file))
            print(f"✅ Saved statistics to: {stats_file}")
            
            # ════════════════════════════════════════════════════════════════
            # SUMMARY
            # ════════════════════════════════════════════════════════════════
            
            print(f"\n{'='*80}")
            print("PIPELINE COMPLETE")
            print(f"{'='*80}")
            
            print(f"\n📊 STATISTICS:")
            print(f"   Seed Queries: {stats['total_seeds']}")
            print(f"   Persona Variations: {stats['total_personas']}")
            print(f"   Complexity Variations: {stats['total_complexities']}")
            print(f"   Training Examples Generated: {stats['total_training_examples']}")
            
            avg_examples = stats['total_training_examples'] / stats['total_seeds']
            print(f"\n   Average Examples per Seed: {avg_examples:.1f}")
            
            print(f"\n🎯 EXPANSION FACTOR:")
            print(f"   Achieved: ~{avg_examples:.0f}× per seed query")
            print(f"   (With full transformations: 15-45× possible)")
            
            print(f"\n✅ OUTPUT:")
            print(f"   Training Data: {output_file}")
            print(f"   Statistics: {stats_file}")
            
            if stats["errors"]:
                print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
                print(f"   (See {stats_file} for details)")
            
            print(f"\n{'='*80}")
            
            return 0
        
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            
            # Save partial stats
            stats["end_time"] = datetime.now().isoformat()
            stats["errors"].append(str(e))
            stats_file = output_dir / "pipeline_stats_FAILED.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            return 1
    
    def _ingest_pdfs(self, pdf_dir: str) -> Dict[str, Any]:
        """
        Ingest PDFs from directory.
        
        Returns:
            Dictionary with results
        """
        pdf_dir_path = Path(pdf_dir)
        pdf_files = list(pdf_dir_path.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files")
        print("Ingesting PDFs...\n")
        
        # Initialize components
        provider = BedrockProvider(
            model_id=self.config.bedrock.model_id,
            embedding_model_id=self.config.bedrock.embedding_model_id,
            region=self.config.bedrock.region
        )
        
        parser = PDFParser(
            chunk_size=self.config.pdf_processing.chunk_size,
            chunk_overlap=self.config.pdf_processing.chunk_overlap,
            extract_images=self.config.pdf_processing.extract_images
        )
        
        db_manager = ChromaDBManager(
            persist_directory=self.config.chromadb.persist_directory,
            collection_name=self.config.chromadb.collection_name
        )
        
        # Process PDFs
        success_count = 0
        errors = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"  [{i}/{len(pdf_files)}] {pdf_file.name}...")
            
            try:
                result = parser.parse_pdf(
                    str(pdf_file),
                    vision_provider=provider,
                    analyze_images=self.config.pdf_processing.use_vision_for_images
                )
                
                documents, metadatas = parser.chunks_to_documents(result['chunks'])
                embeddings = provider.generate_embeddings_batch(documents)
                
                db_manager.add_documents(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                success_count += 1
                print(f"    ✅ Success ({len(result['chunks'])} chunks)\n")
            
            except Exception as e:
                error_msg = f"Failed to ingest {pdf_file.name}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                print(f"    ❌ Failed: {e}\n")
        
        print(f"✅ Ingested {success_count}/{len(pdf_files)} PDFs")
        
        return {
            "exit_code": 0 if success_count == len(pdf_files) else 1,
            "success_count": success_count,
            "total_pdfs": len(pdf_files),
            "errors": errors
        }
