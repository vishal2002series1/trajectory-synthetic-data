"""
Pipeline commands for end-to-end workflow orchestration.

Handles:
- Full pipeline: Ingest → Transform → Generate
- Checkpoint and resume functionality
- Comprehensive progress tracking
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..core import BedrockProvider, VectorStore, PDFParser, ChromaDBManager
from ..transformations import PersonaTransformer, QueryModifier, ToolDataTransformer
from ..generators import TrajectoryGeneratorV2
from ..utils import get_logger, ensure_dir, write_jsonl, read_json

logger = get_logger(__name__)


class PipelineCommand:
    """Command handler for end-to-end pipeline execution."""
    
    def __init__(self, config):
        """
        Initialize pipeline command.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info("PipelineCommand initialized")
    
    def run_pipeline(
        self,
        seed_file: str,
        pdf_dir: Optional[str] = None,
        skip_ingest: bool = False,
        output: Optional[str] = None,
        limit: Optional[int] = None
    ) -> int:
        """
        Run complete end-to-end pipeline.
        
        Pipeline stages:
        1. Ingest PDFs (optional)
        2. Load seed queries
        3. Apply all transformations (30×)
        4. Generate trajectories for each variation
        5. Save training data
        
        Args:
            seed_file: Path to seed queries file
            pdf_dir: Directory containing PDFs to ingest
            skip_ingest: Skip ingestion step
            output: Output directory
            limit: Maximum number of training examples
            
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
        print("COMPLETE PIPELINE EXECUTION")
        print(f"{'='*80}")
        print(f"\n📋 Pipeline Configuration:")
        print(f"   Seed file: {seed_path.name}")
        if pdf_dir and not skip_ingest:
            print(f"   PDF directory: {pdf_dir}")
        else:
            print(f"   PDF ingestion: SKIPPED (using existing ChromaDB)")
        print(f"   Output directory: {output_dir}")
        if limit:
            print(f"   Limit: {limit} training examples")
        
        print(f"\n📊 Expected Expansion:")
        print(f"   Per seed query: 30× (5 personas × 3 complexity × 2 tool data)")
        
        # Prompt for confirmation
        response = input(f"\nProceed with pipeline? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Pipeline cancelled")
            return 1
        
        # Track statistics
        stats = {
            "start_time": datetime.now().isoformat(),
            "seed_queries": 0,
            "pdfs_ingested": 0,
            "transformations_generated": 0,
            "trajectories_generated": 0,
            "training_examples": 0,
            "errors": []
        }
        
        try:
            # ================================================================
            # STAGE 1: PDF INGESTION (Optional)
            # ================================================================
            
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
            
            # ================================================================
            # STAGE 2: INITIALIZE COMPONENTS
            # ================================================================
            
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
            tool_tx = ToolDataTransformer(provider)
            print("✅ Transformers initialized")
            
            # Initialize trajectory generator
            generator = TrajectoryGeneratorV2(
                bedrock_provider=provider,
                vector_store=vector_store,
                config=self.config
            )
            print("✅ Trajectory generator initialized")
            
            # ================================================================
            # STAGE 3: LOAD SEED QUERIES
            # ================================================================
            
            print(f"\n{'='*80}")
            print("STAGE 3: LOADING SEED QUERIES")
            print(f"{'='*80}\n")
            
            seed_data = read_json(str(seed_path))
            
            if isinstance(seed_data, dict) and 'seeds' in seed_data:
                seed_queries = [item['query'] for item in seed_data['seeds']]
            elif isinstance(seed_data, list):
                seed_queries = [item['query'] if isinstance(item, dict) else item 
                               for item in seed_data]
            else:
                print(f"❌ Error: Invalid seed file format")
                return 1
            
            stats["seed_queries"] = len(seed_queries)
            print(f"✅ Loaded {len(seed_queries)} seed queries")
            
            # ================================================================
            # STAGE 4: TRANSFORMATION PIPELINE
            # ================================================================
            
            print(f"\n{'='*80}")
            print("STAGE 4: TRANSFORMATION PIPELINE (30× EXPANSION)")
            print(f"{'='*80}\n")
            
            all_transformations = []
            
            for seed_idx, seed_query in enumerate(seed_queries, 1):
                print(f"\n{'─'*80}")
                print(f"SEED {seed_idx}/{len(seed_queries)}: {seed_query[:60]}...")
                print(f"{'─'*80}\n")
                
                try:
                    # Persona transformation (×5)
                    print("  Step 1: Persona transformation...")
                    personas = persona_tx.transform(seed_query)
                    print(f"    ✅ Generated {len(personas)} variations\n")
                    
                    # For each persona, apply query modification
                    for persona_code, persona_query in personas.items():
                        print(f"  Step 2.{persona_code}: Query modification...")
                        complexities = query_mod.transform(persona_query, include_original=True)
                        print(f"    ✅ Generated {len(complexities)} variations\n")
                        
                        # For each complexity, apply tool data transformation
                        for complexity, complex_query in complexities.items():
                            print(f"  Step 3.{persona_code}.{complexity}: Tool data...")
                            tool_vars = tool_tx.transform(
                                query=complex_query,
                                tools_used=["search_knowledge_base"],
                                correct_answer="Sample answer"
                            )
                            print(f"    ✅ Generated {len(tool_vars)} variations\n")
                            
                            # Store each variation
                            for tool_type, tool_var in tool_vars.items():
                                all_transformations.append({
                                    "seed_query": seed_query,
                                    "seed_index": seed_idx,
                                    "persona": persona_code,
                                    "complexity": complexity,
                                    "tool_data_type": tool_type,
                                    "transformed_query": complex_query,
                                    "tool_data": tool_var.tool_data,
                                    "expected_behavior": tool_var.expected_behavior
                                })
                
                except Exception as e:
                    error_msg = f"Transformation failed for seed {seed_idx}: {e}"
                    print(f"    ❌ {error_msg}\n")
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
            
            stats["transformations_generated"] = len(all_transformations)
            
            print(f"\n{'─'*80}")
            print(f"✅ Transformation stage complete")
            print(f"   Generated {len(all_transformations)} transformed queries")
            print(f"{'─'*80}")
            
            # Save transformations
            transform_file = output_dir / "transformations.jsonl"
            write_jsonl(all_transformations, str(transform_file))
            print(f"💾 Saved transformations: {transform_file}")
            
            # Apply limit if specified
            if limit:
                all_transformations = all_transformations[:limit]
                print(f"\n⚙️  Limited to {len(all_transformations)} transformations")
            
            # ================================================================
            # STAGE 5: TRAJECTORY GENERATION
            # ================================================================
            
            print(f"\n{'='*80}")
            print("STAGE 5: TRAJECTORY GENERATION")
            print(f"{'='*80}\n")
            
            training_examples = []
            
            for idx, transformation in enumerate(all_transformations, 1):
                print(f"[{idx}/{len(all_transformations)}] Generating trajectory...")
                print(f"  Query: {transformation['transformed_query'][:70]}...")
                
                try:
                    trajectory = generator.generate_trajectory(
                        query=transformation['transformed_query'],
                        n_results=3,
                        abstract=True
                    )
                    
                    # Convert to training format
                    training_example = generator.trajectory_to_output_format(
                        trajectory=trajectory,
                        include_metadata=True,
                        include_tool_results=False
                    )
                    
                    # Add transformation metadata
                    training_example['metadata'].update({
                        "seed_query": transformation['seed_query'],
                        "persona": transformation['persona'],
                        "complexity": transformation['complexity'],
                        "tool_data_type": transformation['tool_data_type']
                    })
                    
                    training_examples.append(training_example)
                    print(f"  ✅ Generated\n")
                
                except Exception as e:
                    error_msg = f"Trajectory generation failed for item {idx}: {e}"
                    print(f"  ❌ {error_msg}\n")
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
            
            stats["trajectories_generated"] = len(training_examples)
            stats["training_examples"] = len(training_examples)
            
            print(f"\n{'─'*80}")
            print(f"✅ Trajectory generation complete")
            print(f"   Generated {len(training_examples)} training examples")
            print(f"{'─'*80}")
            
            # ================================================================
            # STAGE 6: SAVE TRAINING DATA
            # ================================================================
            
            print(f"\n{'='*80}")
            print("STAGE 6: SAVING TRAINING DATA")
            print(f"{'='*80}\n")
            
            # Save training data
            training_file = output_dir / "training_data.jsonl"
            write_jsonl(training_examples, str(training_file))
            print(f"💾 Saved training data: {training_file}")
            
            # Save statistics
            stats["end_time"] = datetime.now().isoformat()
            stats_file = output_dir / "pipeline_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"💾 Saved statistics: {stats_file}")
            
            # ================================================================
            # FINAL SUMMARY
            # ================================================================
            
            print(f"\n{'='*80}")
            print("PIPELINE COMPLETE")
            print(f"{'='*80}")
            
            print(f"\n📊 Final Statistics:")
            print(f"   Seed queries: {stats['seed_queries']}")
            if stats["pdfs_ingested"] > 0:
                print(f"   PDFs ingested: {stats['pdfs_ingested']}")
            print(f"   Transformations: {stats['transformations_generated']}")
            print(f"   Trajectories: {stats['trajectories_generated']}")
            print(f"   Training examples: {stats['training_examples']}")
            
            if stats["errors"]:
                print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
                print(f"   (See {stats_file} for details)")
            
            print(f"\n💾 Output directory: {output_dir}")
            print(f"   - transformations.jsonl ({stats['transformations_generated']} items)")
            print(f"   - training_data.jsonl ({stats['training_examples']} items)")
            print(f"   - pipeline_stats.json")
            
            print(f"\n✅ Pipeline executed successfully!")
            
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
