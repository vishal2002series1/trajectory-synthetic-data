#!/usr/bin/env python3
"""
Trajectory Synthetic Data Kit - Main CLI Entry Point

Commands:
  ingest       - Ingest PDF documents into vector database
  transform    - Apply transformation functions to queries
  generate     - Generate synthetic trajectories
  pipeline     - Run complete end-to-end pipeline

Usage:
  python main.py ingest <pdf_path>              # Ingest single PDF
  python main.py ingest-batch <directory>       # Ingest all PDFs in directory
  python main.py transform persona <query>      # Apply persona transformation
  python main.py transform query <query>        # Apply query modification
  python main.py transform tool <query>         # Apply tool data transformation
  python main.py transform all <query>          # Apply all transformations
  python main.py generate <seed_file>           # Generate from seed queries
  python main.py generate --no-seed             # Generate without seeds
  python main.py pipeline <seed_file>           # Full pipeline

Examples:
  python main.py ingest data/pdfs/sample.pdf
  python main.py ingest-batch data/pdfs/
  python main.py transform all "What is my portfolio allocation?"
  python main.py generate data/seed/queries.json
  python main.py pipeline data/seed/queries.json --verbose
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.cli import (
    IngestCommand,
    TransformCommand,
    GenerateCommand,
    PipelineCommand
)
from src.utils import setup_logger, load_config


def create_parser():
    """Create argument parser with all commands."""
    parser = argparse.ArgumentParser(
        description="Trajectory Synthetic Data Kit - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global options
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================================================
    # INGEST COMMANDS
    # =========================================================================
    
    # Single PDF ingest
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Ingest a single PDF document'
    )
    ingest_parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to PDF file'
    )
    ingest_parser.add_argument(
        '--skip-vision',
        action='store_true',
        help='Skip vision analysis of images'
    )
    ingest_parser.add_argument(
        '--collection',
        type=str,
        help='ChromaDB collection name (overrides config)'
    )
    
    # Batch PDF ingest
    ingest_batch_parser = subparsers.add_parser(
        'ingest-batch',
        help='Ingest all PDFs in a directory'
    )
    ingest_batch_parser.add_argument(
        'directory',
        type=str,
        help='Directory containing PDF files'
    )
    ingest_batch_parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search for PDFs recursively in subdirectories'
    )
    ingest_batch_parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    ingest_batch_parser.add_argument(
        '--skip-vision',
        action='store_true',
        help='Skip vision analysis of images'
    )
    ingest_batch_parser.add_argument(
        '--collection',
        type=str,
        help='ChromaDB collection name (overrides config)'
    )
    
    # =========================================================================
    # TRANSFORM COMMANDS
    # =========================================================================
    
    transform_parser = subparsers.add_parser(
        'transform',
        help='Apply transformation functions'
    )
    transform_subparsers = transform_parser.add_subparsers(
        dest='transform_type',
        help='Transformation type'
    )
    
    # Persona transformation
    persona_parser = transform_subparsers.add_parser(
        'persona',
        help='Apply persona transformation'
    )
    persona_parser.add_argument(
        'query',
        type=str,
        help='Query to transform'
    )
    persona_parser.add_argument(
        '--personas',
        type=str,
        nargs='+',
        help='Specific personas to use (default: all)'
    )
    persona_parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: stdout)'
    )
    
    # Query modification
    query_parser = transform_subparsers.add_parser(
        'query',
        help='Apply query complexity modification'
    )
    query_parser.add_argument(
        'query',
        type=str,
        help='Query to modify'
    )
    query_parser.add_argument(
        '--include-original',
        action='store_true',
        help='Include original query in output'
    )
    query_parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: stdout)'
    )
    
    # Tool data transformation
    tool_parser = transform_subparsers.add_parser(
        'tool',
        help='Apply tool data transformation'
    )
    tool_parser.add_argument(
        'query',
        type=str,
        help='Query to transform'
    )
    tool_parser.add_argument(
        '--tools',
        type=str,
        nargs='+',
        help='Tools used (e.g., search_knowledge_base get_allocation)'
    )
    tool_parser.add_argument(
        '--answer',
        type=str,
        help='Correct answer for the query'
    )
    tool_parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: stdout)'
    )
    
    # All transformations
    all_parser = transform_subparsers.add_parser(
        'all',
        help='Apply all transformation functions'
    )
    all_parser.add_argument(
        'query',
        type=str,
        help='Query to transform'
    )
    all_parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results (default: data/output/transformations/YYYYMMDD_HHMMSS)'
    )
    all_parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'jsonl', 'both'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    
    # =========================================================================
    # GENERATE COMMANDS
    # =========================================================================
    
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate synthetic trajectories'
    )
    generate_parser.add_argument(
        'seed_file',
        type=str,
        nargs='?',
        help='Path to seed queries JSON file (optional if --no-seed)'
    )
    generate_parser.add_argument(
        '--no-seed',
        action='store_true',
        help='Generate queries without seed file'
    )
    generate_parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: data/output/trajectories/YYYYMMDD_HHMMSS)'
    )
    generate_parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of examples to generate'
    )
    generate_parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'jsonl', 'both'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    
    # =========================================================================
    # PIPELINE COMMAND
    # =========================================================================
    
    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Run complete end-to-end pipeline'
    )
    pipeline_parser.add_argument(
        'seed_file',
        type=str,
        help='Path to seed queries JSON file'
    )
    pipeline_parser.add_argument(
        '--pdf-dir',
        type=str,
        help='Directory containing PDFs to ingest (optional)'
    )
    pipeline_parser.add_argument(
        '--skip-ingest',
        action='store_true',
        help='Skip ingestion step (use existing ChromaDB)'
    )
    pipeline_parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: data/output/pipeline/YYYYMMDD_HHMMSS)'
    )
    pipeline_parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of training examples to generate'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 0
    
    # Determine log level
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    # Setup logging
    setup_logger(log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Route to appropriate command handler
    try:
        if args.command == 'ingest':
            cmd = IngestCommand(config)
            return cmd.ingest_single(
                pdf_path=args.pdf_path,
                skip_vision=args.skip_vision,
                collection=args.collection
            )
        
        elif args.command == 'ingest-batch':
            cmd = IngestCommand(config)
            return cmd.ingest_batch(
                directory=args.directory,
                recursive=args.recursive,
                parallel=args.parallel,
                skip_vision=args.skip_vision,
                collection=args.collection
            )
        
        elif args.command == 'transform':
            cmd = TransformCommand(config)
            
            if args.transform_type == 'persona':
                return cmd.transform_persona(
                    query=args.query,
                    personas=args.personas,
                    output=args.output
                )
            
            elif args.transform_type == 'query':
                return cmd.transform_query(
                    query=args.query,
                    include_original=args.include_original,
                    output=args.output
                )
            
            elif args.transform_type == 'tool':
                return cmd.transform_tool(
                    query=args.query,
                    tools=args.tools,
                    answer=args.answer,
                    output=args.output
                )
            
            elif args.transform_type == 'all':
                return cmd.transform_all(
                    query=args.query,
                    output=args.output,
                    format=args.format
                )
            
            else:
                print(f"❌ Unknown transform type: {args.transform_type}")
                return 1
        
        elif args.command == 'generate':
            cmd = GenerateCommand(config)
            return cmd.generate(
                seed_file=args.seed_file,
                no_seed=args.no_seed,
                output=args.output,
                limit=args.limit,
                format=args.format
            )
        
        elif args.command == 'pipeline':
            cmd = PipelineCommand(config)
            return cmd.run_pipeline(
                seed_file=args.seed_file,
                pdf_dir=args.pdf_dir,
                skip_ingest=args.skip_ingest,
                output=args.output,
                limit=args.limit
            )
        
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 130
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
