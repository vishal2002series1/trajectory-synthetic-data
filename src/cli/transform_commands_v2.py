"""
Transform commands for query transformation operations.

Handles:
- Persona transformation (×5)
- Query complexity modification (×3)
- Tool data transformation (×2)
- Combined transformations (all at once)
"""

import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from ..core import BedrockProvider
from ..transformations import PersonaTransformer, QueryModifier, ToolDataTransformer
from ..utils import get_logger, ensure_dir, write_json, write_jsonl

logger = get_logger(__name__)


class TransformCommand:
    """Command handler for query transformations."""
    
    def __init__(self, config):
        """
        Initialize transform command.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info("TransformCommand initialized")
    
    def transform_persona(
        self,
        query: str,
        personas: Optional[List[str]] = None,
        output: Optional[str] = None
    ) -> int:
        """
        Apply persona transformation.
        
        Args:
            query: Query to transform
            personas: Specific personas to use
            output: Output file path
            
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        print(f"\n{'='*80}")
        print("PERSONA TRANSFORMATION")
        print(f"{'='*80}")
        print(f"\n📝 Original Query:")
        print(f"   {query}")
        
        try:
            # Initialize provider and transformer
            provider = BedrockProvider(
                model_id=self.config.bedrock.model_id,
                region=self.config.bedrock.region
            )
            
            transformer = PersonaTransformer(provider)
            
            # Transform
            print(f"\n⚙️  Applying persona transformation...")
            print(f"   Personas: {personas if personas else 'All (5)'}")
            
            variations = transformer.transform(query, personas=personas)
            
            print(f"\n✅ Generated {len(variations)} persona variations\n")
            
            # Display results
            print(f"{'─'*80}")
            print("RESULTS")
            print(f"{'─'*80}\n")
            
            for persona_code, transformed_query in variations.items():
                persona_info = transformer.get_persona_info(persona_code)
                persona_name = persona_info.name if persona_info else persona_code
                
                print(f"{persona_code} ({persona_name}):")
                print(f"  {transformed_query}\n")
            
            # Save to file if requested
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                output_data = {
                    "original_query": query,
                    "variations": variations,
                    "timestamp": datetime.now().isoformat()
                }
                
                write_json(output_data, str(output_path))
                print(f"💾 Saved to: {output_path}")
            
            return 0
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Persona transformation failed: {e}", exc_info=True)
            return 1
    
    def transform_query(
        self,
        query: str,
        include_original: bool = False,
        output: Optional[str] = None
    ) -> int:
        """
        Apply query complexity modification.
        
        Args:
            query: Query to modify
            include_original: Include original query
            output: Output file path
            
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        print(f"\n{'='*80}")
        print("QUERY COMPLEXITY MODIFICATION")
        print(f"{'='*80}")
        print(f"\n📝 Original Query:")
        print(f"   {query}")
        
        try:
            # Initialize provider and modifier
            provider = BedrockProvider(
                model_id=self.config.bedrock.model_id,
                region=self.config.bedrock.region
            )
            
            modifier = QueryModifier(provider)
            
            # Transform
            print(f"\n⚙️  Applying complexity modification...")
            print(f"   Include original: {include_original}")
            
            variations = modifier.transform(query, include_original=include_original)
            
            print(f"\n✅ Generated {len(variations)} complexity variations\n")
            
            # Display results
            print(f"{'─'*80}")
            print("RESULTS")
            print(f"{'─'*80}\n")
            
            complexity_labels = {
                "Q-": "Simplified",
                "Q": "Original",
                "Q+": "Complex"
            }
            
            for complexity, modified_query in variations.items():
                label = complexity_labels.get(complexity, complexity)
                print(f"{complexity} ({label}):")
                print(f"  {modified_query}\n")
            
            # Save to file if requested
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                output_data = {
                    "original_query": query,
                    "variations": variations,
                    "timestamp": datetime.now().isoformat()
                }
                
                write_json(output_data, str(output_path))
                print(f"💾 Saved to: {output_path}")
            
            return 0
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Query modification failed: {e}", exc_info=True)
            return 1
    
    def transform_tool(
        self,
        query: str,
        tools: Optional[List[str]] = None,
        answer: Optional[str] = None,
        output: Optional[str] = None
    ) -> int:
        """
        Apply tool data transformation.
        
        Args:
            query: Query to transform
            tools: Tools used
            answer: Correct answer
            output: Output file path
            
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        print(f"\n{'='*80}")
        print("TOOL DATA TRANSFORMATION")
        print(f"{'='*80}")
        print(f"\n📝 Query: {query}")
        print(f"   Tools: {tools if tools else 'None specified'}")
        print(f"   Answer: {answer[:100] if answer else 'None specified'}...")
        
        try:
            # Initialize provider and transformer
            provider = BedrockProvider(
                model_id=self.config.bedrock.model_id,
                region=self.config.bedrock.region
            )
            
            transformer = ToolDataTransformer(provider)
            
            # Transform
            print(f"\n⚙️  Applying tool data transformation...")
            
            variations = transformer.transform(
                query=query,
                tools_used=tools or [],
                correct_answer=answer or ""
            )
            
            print(f"\n✅ Generated {len(variations)} tool data variations\n")
            
            # Display results
            print(f"{'─'*80}")
            print("RESULTS")
            print(f"{'─'*80}\n")
            
            for var_type, variation in variations.items():
                print(f"{var_type.upper()}:")
                print(f"  Data Type: {variation.data_type}")
                print(f"  Expected Behavior: {variation.expected_behavior}")
                print(f"  Tool Data: {json.dumps(variation.tool_data, indent=4)}")
                print(f"  Decision: {variation.decision[:150]}...")
                print()
            
            # Save to file if requested
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                output_data = {
                    "query": query,
                    "tools_used": tools,
                    "correct_answer": answer,
                    "variations": {
                        k: {
                            "data_type": v.data_type,
                            "expected_behavior": v.expected_behavior,
                            "tool_data": v.tool_data,
                            "decision": v.decision
                        }
                        for k, v in variations.items()
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                write_json(output_data, str(output_path))
                print(f"💾 Saved to: {output_path}")
            
            return 0
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Tool data transformation failed: {e}", exc_info=True)
            return 1
    
    def transform_all(
        self,
        query: str,
        output: Optional[str] = None,
        format: str = 'jsonl'
    ) -> int:
        """
        Apply all transformations.
        
        This generates the full 30× expansion:
        - 5 personas × 3 complexity levels × 2 tool data variants = 30
        
        Args:
            query: Query to transform
            output: Output directory
            format: Output format (json, jsonl, both)
            
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        print(f"\n{'='*80}")
        print("ALL TRANSFORMATIONS (30× EXPANSION)")
        print(f"{'='*80}")
        print(f"\n📝 Original Query:")
        print(f"   {query}")
        print(f"\n⚙️  Applying:")
        print(f"   - Persona transformation (×5)")
        print(f"   - Query complexity modification (×3 per persona)")
        print(f"   - Tool data transformation (×2 per complexity)")
        print(f"   = 5 × 3 × 2 = 30 variations")
        
        # Prompt for confirmation
        response = input(f"\nThis will generate 30 variations. Proceed? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Transformation cancelled")
            return 1
        
        try:
            # Initialize components
            print(f"\n{'─'*80}")
            print("Initializing transformers...")
            print(f"{'─'*80}\n")
            
            provider = BedrockProvider(
                model_id=self.config.bedrock.model_id,
                region=self.config.bedrock.region
            )
            
            persona_tx = PersonaTransformer(provider)
            query_mod = QueryModifier(provider)
            tool_tx = ToolDataTransformer(provider)
            
            print("✅ All transformers initialized\n")
            
            # Apply transformations
            print(f"{'─'*80}")
            print("TRANSFORMATION PIPELINE")
            print(f"{'─'*80}\n")
            
            all_variations = []
            variation_count = 0
            
            # Step 1: Persona transformation
            print("Step 1: Persona transformation...")
            personas = persona_tx.transform(query)
            print(f"   ✅ Generated {len(personas)} persona variations\n")
            
            # Step 2: For each persona, apply query modification
            for persona_code, persona_query in personas.items():
                print(f"Step 2.{persona_code}: Query modification for {persona_code}...")
                complexities = query_mod.transform(persona_query, include_original=True)
                print(f"   ✅ Generated {len(complexities)} complexity variations\n")
                
                # Step 3: For each complexity, apply tool data transformation
                for complexity, complex_query in complexities.items():
                    print(f"Step 3.{persona_code}.{complexity}: Tool data transformation...")
                    tool_variations = tool_tx.transform(
                        query=complex_query,
                        tools_used=["search_knowledge_base"],
                        correct_answer="Sample answer"
                    )
                    print(f"   ✅ Generated {len(tool_variations)} tool variations\n")
                    
                    # Store variations
                    for tool_type, tool_var in tool_variations.items():
                        variation_count += 1
                        all_variations.append({
                            "variation_id": variation_count,
                            "original_query": query,
                            "persona": persona_code,
                            "complexity": complexity,
                            "tool_data_type": tool_type,
                            "transformed_query": complex_query,
                            "tool_data": tool_var.tool_data,
                            "expected_behavior": tool_var.expected_behavior,
                            "decision": tool_var.decision
                        })
            
            print(f"{'─'*80}")
            print(f"✅ TRANSFORMATION COMPLETE")
            print(f"{'─'*80}")
            print(f"\nGenerated {len(all_variations)} total variations\n")
            
            # Save results
            if not output:
                # Generate default output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = f"data/output/transformations/{timestamp}"
            
            output_dir = Path(output)
            ensure_dir(output_dir)
            
            if format in ['json', 'both']:
                json_file = output_dir / "transformations.json"
                write_json(all_variations, str(json_file))
                print(f"💾 Saved JSON: {json_file}")
            
            if format in ['jsonl', 'both']:
                jsonl_file = output_dir / "transformations.jsonl"
                write_jsonl(all_variations, str(jsonl_file))
                print(f"💾 Saved JSONL: {jsonl_file}")
            
            # Summary
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            print(f"\n📊 Transformation Statistics:")
            print(f"   Original query: 1")
            print(f"   Persona variations: {len(personas)}")
            print(f"   Complexity variations: {len(complexities)} per persona")
            print(f"   Tool data variations: 2 per complexity")
            print(f"   Total variations: {len(all_variations)}")
            print(f"\n💾 Output saved to: {output_dir}")
            
            return 0
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"All transformations failed: {e}", exc_info=True)
            return 1
