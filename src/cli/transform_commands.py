"""
Transform Commands for CLI - CORRECTED VERSION

Outputs transformations in the format specified in config.yaml:
- Qi (query field name from config)
- Not "transformed_query" or other custom fields
"""

import typer
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

from ..core import BedrockProvider
from ..transformations import PersonaTransformer, QueryModifier, ToolDataTransformer  
from ..utils import load_config, get_logger, write_jsonl

logger = get_logger(__name__)


class TransformCommand:
    """Handle transformation commands."""
    
    def __init__(self, config=None):
        """
        Initialize transform command.
        
        Args:
            config: Optional config object (loaded automatically if not provided)
        """
        self.config = config if config is not None else load_config()
        self.provider = None
        self.persona_tx = None
        self.query_mod = None
        self.tool_tx = None
        logger.info("TransformCommand initialized")
    
    def _initialize_transformers(self):
        """Initialize transformation components."""
        if self.persona_tx is not None:
            return  # Already initialized
        
        print("─" * 80)
        print("Initializing Transformers")
        print("─" * 80)
        
        # Initialize Bedrock Provider
        self.provider = BedrockProvider(
            model_id=self.config.bedrock.model_id,
            region=self.config.bedrock.region
        )
        print("✅ Bedrock provider initialized")
        
        # Initialize transformers
        self.persona_tx = PersonaTransformer(self.provider)
        self.query_mod = QueryModifier(self.provider)
        self.tool_tx = ToolDataTransformer(self.provider)
        print("✅ Transformers initialized")
        print(f"   - Persona: {self.persona_tx.get_expansion_factor()}× expansion")
        print(f"   - Query complexity: {self.query_mod.get_expansion_factor()}× expansion")
        print(f"   - Tool data: {self.tool_tx.get_expansion_factor()}× expansion")
    
    def _format_transformation_for_output(
        self,
        original_query: str,
        transformed_query: str,
        persona: str,
        complexity: str,
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Format transformation in config.yaml format.
        
        Uses the field name specified in config (default: "Qi") instead of "transformed_query".
        """
        # Get the query field name from config
        query_field = self.config.output.schema.fields.query  # Default: "Qi"
        
        output = {
            query_field: transformed_query,  # Use config field name!
            "metadata": {
                "original_query": original_query,
                "persona": persona,
                "complexity": complexity,
                "transformation_type": "query_transformation"
            }
        }
        
        if metadata:
            output["metadata"].update(metadata)
        
        return output
    
    def transform_all(
        self,
        query: str,
        output: Optional[str] = None,
        format: str = "jsonl"
    ):
        """
        Apply all transformations to a single query.
        
        Generates 30 variations: 5 personas × 3 complexity × 2 tool data variants
        
        Args:
            query: Input query to transform
            output: Custom output directory (optional)
            format: Output format (jsonl or json)
        """
        print("\n" + "=" * 80)
        print("APPLY ALL TRANSFORMATIONS")
        print("=" * 80)
        print(f"\nOriginal query: {query}")
        
        # Initialize transformers
        self._initialize_transformers()
        
        print("\n" + "─" * 80)
        print("Generating Transformations")
        print("─" * 80)
        
        all_transformations = []
        
        # Step 1: Persona transformation
        print("\n1. Applying Persona Transformation...")
        personas = self.persona_tx.transform(query)
        print(f"   ✅ Generated {len(personas)} persona variations")
        
        # Step 2: For each persona, apply query complexity
        for persona_code, persona_query in personas.items():
            print(f"\n2. Processing persona {persona_code}...")
            
            # Apply query complexity modifications
            complexities = self.query_mod.transform(persona_query, include_original=True)
            print(f"   ✅ Generated {len(complexities)} complexity variations")
            
            # Add each variation in proper format
            for complexity, complex_query in complexities.items():
                # Format using config.yaml field names
                transformation = self._format_transformation_for_output(
                    original_query=query,
                    transformed_query=complex_query,
                    persona=persona_code,
                    complexity=complexity
                )
                all_transformations.append(transformation)
        
        print(f"\n✅ Total transformations: {len(all_transformations)}")
        
        # Save results
        print("\n" + "─" * 80)
        print("Saving Results")
        print("─" * 80)
        
        # Determine output directory
        if output:
            out_dir = Path(output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(self.config.output.output_dir) / "transformations" / timestamp
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename based on format
        if format == "jsonl":
            output_file = out_dir / "transformations.jsonl"
        elif format == "json":
            output_file = out_dir / "transformations.json"
        else:
            output_file = out_dir / "transformations.jsonl"  # Default to jsonl
        
        # Save using appropriate format
        if format == "json":
            # Save as single JSON array
            import json
            with open(output_file, 'w') as f:
                json.dump(all_transformations, f, indent=2)
        else:
            # Save as JSONL (default)
            write_jsonl(all_transformations, output_file)
        
        print(f"✅ Saved {len(all_transformations)} transformations to:")
        print(f"   {output_file}")
        
        # Show sample
        print("\n" + "─" * 80)
        print("Sample Output (First 2)")
        print("─" * 80)
        
        query_field = self.config.output.schema.fields.query
        
        for i, trans in enumerate(all_transformations[:2], 1):
            print(f"\n{i}. {trans['metadata']['persona']} + {trans['metadata']['complexity']}")
            print(f"   {query_field}: {trans[query_field][:80]}...")
        
        # Summary
        print("\n" + "=" * 80)
        print("TRANSFORMATION COMPLETE")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   Input: 1 query")
        print(f"   Output: {len(all_transformations)} transformed queries")
        print(f"   Format: {query_field} (from config.yaml)")
        print(f"   File: {output_file}")
        print(f"\n💡 Next step: Run 'python main.py generate {output_file}'")
    
    def transform_persona(
        self,
        query: str,
        output: Optional[str] = None,
        format: str = "jsonl"
    ):
        """
        Apply only persona transformations.
        
        Args:
            query: Input query
            output: Custom output directory (optional)
            format: Output format (jsonl or json)
        """
        print("\n" + "=" * 80)
        print("PERSONA TRANSFORMATION")
        print("=" * 80)
        print(f"\nOriginal query: {query}")
        
        # Initialize transformers
        self._initialize_transformers()
        
        print("\n" + "─" * 80)
        print("Generating Persona Variations")
        print("─" * 80)
        
        # Generate personas
        personas = self.persona_tx.transform(query)
        print(f"✅ Generated {len(personas)} persona variations")
        
        # Format for output
        transformations = []
        for persona_code, persona_query in personas.items():
            transformation = self._format_transformation_for_output(
                original_query=query,
                transformed_query=persona_query,
                persona=persona_code,
                complexity="Q"  # Original complexity
            )
            transformations.append(transformation)
        
        # Save results
        if output:
            out_dir = Path(output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(self.config.output.output_dir) / "transformations" / f"persona_{timestamp}"
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename based on format
        if format == "jsonl":
            output_file = out_dir / "persona_transformations.jsonl"
        elif format == "json":
            output_file = out_dir / "persona_transformations.json"
        else:
            output_file = out_dir / "persona_transformations.jsonl"
        
        # Save using appropriate format
        if format == "json":
            import json
            with open(output_file, 'w') as f:
                json.dump(transformations, f, indent=2)
        else:
            write_jsonl(transformations, output_file)
        
        print(f"\n✅ Saved to: {output_file}")
        print(f"   Format: {self.config.output.schema.fields.query} field (config.yaml)")
    
    def transform_complexity(
        self,
        query: str,
        output: Optional[str] = None,
        format: str = "jsonl"
    ):
        """
        Apply only complexity transformations.
        
        Args:
            query: Input query
            output: Custom output directory (optional)
            format: Output format (jsonl or json)
        """
        print("\n" + "=" * 80)
        print("COMPLEXITY TRANSFORMATION")
        print("=" * 80)
        print(f"\nOriginal query: {query}")
        
        # Initialize transformers
        self._initialize_transformers()
        
        print("\n" + "─" * 80)
        print("Generating Complexity Variations")
        print("─" * 80)
        
        # Generate complexities
        complexities = self.query_mod.transform(query, include_original=True)
        print(f"✅ Generated {len(complexities)} complexity variations")
        
        # Format for output
        transformations = []
        for complexity, complex_query in complexities.items():
            transformation = self._format_transformation_for_output(
                original_query=query,
                transformed_query=complex_query,
                persona="P1",  # Default persona
                complexity=complexity
            )
            transformations.append(transformation)
        
        # Save results
        if output:
            out_dir = Path(output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(self.config.output.output_dir) / "transformations" / f"complexity_{timestamp}"
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename based on format
        if format == "jsonl":
            output_file = out_dir / "complexity_transformations.jsonl"
        elif format == "json":
            output_file = out_dir / "complexity_transformations.json"
        else:
            output_file = out_dir / "complexity_transformations.jsonl"
        
        # Save using appropriate format
        if format == "json":
            import json
            with open(output_file, 'w') as f:
                json.dump(transformations, f, indent=2)
        else:
            write_jsonl(transformations, output_file)
        
        print(f"\n✅ Saved to: {output_file}")
        print(f"   Format: {self.config.output.schema.fields.query} field (config.yaml)")


def create_transform_app() -> typer.Typer:
    """Create transform command group."""
    app = typer.Typer(help="Transform queries with persona/complexity variations")
    cmd = TransformCommand()
    
    @app.command("all")
    def transform_all(
        query: str = typer.Argument(..., help="Query to transform"),
        output: Optional[str] = typer.Option(None, help="Custom output directory"),
        format: str = typer.Option("jsonl", help="Output format: jsonl or json")
    ):
        """Apply all transformations (persona × complexity)."""
        cmd.transform_all(query, output, format)
    
    @app.command("persona")
    def transform_persona(
        query: str = typer.Argument(..., help="Query to transform"),
        output: Optional[str] = typer.Option(None, help="Custom output directory"),
        format: str = typer.Option("jsonl", help="Output format: jsonl or json")
    ):
        """Apply only persona transformations."""
        cmd.transform_persona(query, output, format)
    
    @app.command("complexity")
    def transform_complexity(
        query: str = typer.Argument(..., help="Query to transform"),
        output: Optional[str] = typer.Option(None, help="Custom output directory"),
        format: str = typer.Option("jsonl", help="Output format: jsonl or json")
    ):
        """Apply only complexity transformations."""
        cmd.transform_complexity(query, output, format)
    
    return app