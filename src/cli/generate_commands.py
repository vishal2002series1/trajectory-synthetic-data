"""
Generate Commands for CLI

Fixed to handle both formats:
1. Simple queries: {"query": "..."}
2. Transformation output: {"transformed_query": "...", "original_query": "...", ...}
"""

import typer
from pathlib import Path
from typing import Optional
import json

from ..core import BedrockProvider, VectorStore
from ..generators import TrajectoryGeneratorV2
from ..utils import load_config, get_logger, read_json, read_jsonl

logger = get_logger(__name__)


class GenerateCommand:
    """Handle trajectory generation commands."""
    
    def __init__(self, config=None):
        """
        Initialize generate command.
        
        Args:
            config: Optional config object (loaded automatically if not provided)
        """
        self.config = config if config is not None else load_config()
        self.provider = None
        self.vector_store = None
        self.generator = None
        logger.info("GenerateCommand initialized")
    
    def _initialize_components(self):
        """Initialize all required components."""
        if self.generator is not None:
            return  # Already initialized
        
        print("─" * 80)
        print("STEP 1: Initializing Components")
        print("─" * 80)
        
        # Initialize VectorStore (it creates its own BedrockProvider and ChromaDB internally)
        self.vector_store = VectorStore(config=self.config)
        doc_count = self.vector_store.count()
        print("✅ VectorStore initialized")
        print(f"   Collection: {self.config.chromadb.collection_name}")
        print(f"   Documents: {doc_count}")
        
        # Get the provider from VectorStore for the generator
        self.provider = self.vector_store.provider
        
        # Initialize Trajectory Generator
        self.generator = TrajectoryGeneratorV2(
            bedrock_provider=self.provider,
            vector_store=self.vector_store,
            config=self.config
        )
        print("✅ Trajectory generator initialized")
    
    def _normalize_query_format(self, item: dict) -> str:
        """
        Normalize different query formats to a standard format.
        
        Reads the query field name from config.yaml and uses that.
        
        Args:
            item: Dictionary from JSONL file
            
        Returns:
            Extracted query string
        """
        # Get the query field name from config.yaml
        query_field = self.config.output.schema.fields.query  # e.g., "Q" or "Qi"
        
        # Priority 1: Check for the configured query field name
        if query_field in item:
            return item[query_field]
        
        # Priority 2: Fallback to common field names for backward compatibility
        if 'transformed_query' in item:
            return item['transformed_query']
        
        if 'query' in item:
            return item['query']
        
        if 'original_query' in item:
            logger.warning("Using 'original_query' as fallback. Consider using transformed queries.")
            return item['original_query']
        
        # If none found, raise error with helpful message
        available_keys = list(item.keys())
        raise KeyError(
            f"Could not find query field. Expected '{query_field}' (from config.yaml). "
            f"Available keys: {available_keys}"
        )
    
    def _load_queries(self, seed_file: str) -> list:
        """
        Load queries from seed file.
        
        Supports both .json and .jsonl formats.
        Automatically detects and normalizes query field names.
        
        Args:
            seed_file: Path to seed file (.json or .jsonl)
            
        Returns:
            List of query strings
        """
        seed_path = Path(seed_file)
        
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_file}")
        
        # Determine file format
        if seed_path.suffix == '.jsonl':
            data = read_jsonl(seed_path)
        elif seed_path.suffix == '.json':
            data = read_json(seed_path)
            # If it's a dict with a 'queries' key, extract that
            if isinstance(data, dict) and 'queries' in data:
                data = data['queries']
            # If it's a single dict, wrap in list
            elif isinstance(data, dict):
                data = [data]
        else:
            raise ValueError(f"Unsupported file format: {seed_path.suffix}. Use .json or .jsonl")
        
        # Extract queries with normalization
        queries = []
        for idx, item in enumerate(data):
            try:
                query = self._normalize_query_format(item)
                queries.append(query)
            except KeyError as e:
                logger.error(f"Error extracting query from item {idx}: {e}")
                logger.error(f"Item content: {item}")
                raise
        
        logger.info(f"Loaded {len(queries)} queries from {seed_file}")
        return queries
    
    def generate(
        self,
        seed_file: str,
        n_results: int = 3,
        abstract: bool = True,
        output: Optional[str] = None,
        no_seed: bool = False,
        limit: Optional[int] = None,
        format: str = "jsonl"  # ADDED format parameter
    ):
        """
        Generate trajectories from seed queries.
        
        Args:
            seed_file: Path to seed file (.json or .jsonl)
            n_results: Number of chunks to retrieve per query
            abstract: Whether to abstract document references
            output: Custom output directory (optional)
            no_seed: Whether to generate without seed queries (not implemented yet)
            limit: Limit number of queries to process (optional)
            format: Output format ('jsonl' or 'json')
        """
        print("\n" + "=" * 80)
        print("TRAJECTORY GENERATION")
        print("=" * 80)
        
        # Check if no_seed mode is requested
        if no_seed:
            print("\n⚠️  No-seed generation is not yet implemented.")
            print("This feature will generate queries directly from PDF content.")
            print("For now, please provide a seed file with queries.")
            return
        
        # Resolve seed file path
        seed_path = Path(seed_file)
        if not seed_path.is_absolute():
            seed_path = Path.cwd() / seed_path
        
        print(f"📄 Seed file: {seed_file}")
        print(f"   Path: {seed_path}")
        print(f"   Output format: {format}")
        
        # Confirm before proceeding
        if not typer.confirm("Proceed with generation?"):
            print("❌ Generation cancelled")
            return
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Load queries
            print("\n" + "─" * 80)
            print("STEP 2: Loading Queries")
            print("─" * 80)
            
            queries = self._load_queries(str(seed_path))
            print(f"✅ Loaded {len(queries)} queries")
            
            # Show first few examples
            print("\n📝 Sample queries:")
            for i, query in enumerate(queries[:3], 1):
                print(f"   {i}. {query[:80]}{'...' if len(query) > 80 else ''}")
            if len(queries) > 3:
                print(f"   ... and {len(queries) - 3} more")
            
            # Apply limit if specified
            if limit is not None and limit > 0:
                queries = queries[:limit]
                print(f"\n⚠️  Limited to first {limit} queries")
            
            # Generate trajectories
            print("\n" + "─" * 80)
            print("STEP 3: Generating Trajectories")
            print("─" * 80)
            
            trajectories = []
            for i, query in enumerate(queries, 1):
                print(f"\n⏳ Generating trajectory {i}/{len(queries)}...")
                print(f"   Query: {query[:80]}{'...' if len(query) > 80 else ''}")
                
                try:
                    trajectory = self.generator.generate_trajectory(
                        query=query,
                        n_results=n_results,
                        abstract=abstract
                    )
                    trajectories.append(trajectory)
                    print(f"   ✅ Generated successfully")
                    
                except Exception as e:
                    logger.error(f"Error generating trajectory {i}: {e}")
                    print(f"   ❌ Error: {e}")
                    continue
            
            print(f"\n✅ Generated {len(trajectories)} trajectories")
            
            # Save results
            print("\n" + "─" * 80)
            print("STEP 4: Saving Results")
            print("─" * 80)
            
            # Determine output directory
            if output:
                out_dir = Path(output)
            else:
                out_dir = Path(self.config.output.output_dir) / "generated"
            
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename based on format
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = format if format in ['json', 'jsonl'] else 'jsonl'
            output_file = out_dir / f"trajectories_{timestamp}.{file_extension}"
            
            # Save trajectories
            self.generator.save_trajectories(
                trajectories=trajectories,
                output_path=str(output_file)
            )
            
            print(f"✅ Saved {len(trajectories)} trajectories to:")
            print(f"   {output_file}")
            
            # Show summary
            print("\n" + "=" * 80)
            print("GENERATION COMPLETE")
            print("=" * 80)
            print(f"\n📊 Summary:")
            print(f"   Input queries: {len(queries)}")
            print(f"   Trajectories generated: {len(trajectories)}")
            print(f"   Success rate: {len(trajectories)/len(queries)*100:.1f}%")
            print(f"   Output format: {format}")
            print(f"   Output file: {output_file}")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            print(f"❌ Error: {e}")
            raise


def create_generate_app() -> typer.Typer:
    """Create generate command group."""
    app = typer.Typer(help="Generate trajectories from queries")
    cmd = GenerateCommand()
    
    @app.command("from-file")
    def generate_from_file(
        seed_file: str = typer.Argument(..., help="Path to seed file (.json or .jsonl)"),
        n_results: int = typer.Option(3, help="Number of chunks to retrieve"),
        abstract: bool = typer.Option(True, help="Abstract document references"),
        output: Optional[str] = typer.Option(None, help="Custom output directory"),
        no_seed: bool = typer.Option(False, help="Generate without seed queries (future feature)"),
        limit: Optional[int] = typer.Option(None, help="Limit number of queries to process"),
        format: str = typer.Option("jsonl", help="Output format (json or jsonl)")  # ADDED format option
    ):
        """Generate trajectories from seed file."""
        cmd.generate(seed_file, n_results, abstract, output, no_seed, limit, format)
    
    return app