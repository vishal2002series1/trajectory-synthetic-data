"""
Persona Variation Transformer

Implements transformation T_persona from the mathematical framework:
T_persona: Q → {P₁(Q), P₂(Q), P₃(Q), P₄(Q), P₅(Q)}

Where:
- P₁ = First-time Investor (novice, explanatory)
- P₂ = Experienced Investor (direct, uses jargon)
- P₃ = Retirement Planner (long-term focused)
- P₄ = High Net-Worth Client (sophisticated, complex)
- P₅ = Urgent Issue Resolver (time-sensitive)

Expansion Factor: 1 → 5 (500% increase)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class PersonaProfile:
    """Represents a persona with its characteristics."""
    name: str
    code: str  # P1, P2, P3, P4, P5
    description: str
    characteristics: List[str]
    example_phrases: List[str]


class PersonaTransformer:
    """
    Transform queries into different persona communication styles.
    
    Maintains semantic intent while varying communication style.
    """
    
    # Persona definitions from mathematical framework
    PERSONAS = {
        "first_time_investor": PersonaProfile(
            name="First-time Investor",
            code="P1",
            description="Novice investor seeking clear, explanatory guidance",
            characteristics=[
                "Uses simple, everyday language",
                "Asks for explanations of terminology",
                "Expresses uncertainty or confusion",
                "Seeks reassurance and step-by-step guidance",
                "Mentions being new to investing"
            ],
            example_phrases=[
                "I'm new to this",
                "Could you explain",
                "I'm not sure I understand",
                "What does that mean?",
                "Can you help me understand"
            ]
        ),
        
        "experienced_investor": PersonaProfile(
            name="Experienced Investor",
            code="P2",
            description="Seasoned investor using direct, technical language",
            characteristics=[
                "Uses financial jargon and technical terms",
                "Direct and concise questions",
                "Assumes knowledge of basic concepts",
                "Focuses on specifics and metrics",
                "Professional tone"
            ],
            example_phrases=[
                "Show me",
                "What's the",
                "Provide breakdown",
                "Calculate",
                "Analyze"
            ]
        ),
        
        "retirement_planner": PersonaProfile(
            name="Retirement Planner",
            code="P3",
            description="Long-term focused, retirement-oriented investor",
            characteristics=[
                "Emphasizes long-term goals",
                "References retirement timeline",
                "Concerned with sustainability",
                "Risk-averse language",
                "Time-horizon focused"
            ],
            example_phrases=[
                "As I plan for retirement",
                "In my retirement portfolio",
                "For long-term stability",
                "Given my age",
                "For the next 20-30 years"
            ]
        ),
        
        "high_net_worth": PersonaProfile(
            name="High Net-Worth Client",
            code="P4",
            description="Sophisticated investor with complex needs",
            characteristics=[
                "Advanced financial terminology",
                "Multi-factor considerations",
                "Tax efficiency focus",
                "Estate planning references",
                "Expects detailed analysis"
            ],
            example_phrases=[
                "Considering tax implications",
                "From an estate planning perspective",
                "Optimal allocation for",
                "Factor in",
                "Comprehensive analysis"
            ]
        ),
        
        "urgent_resolver": PersonaProfile(
            name="Urgent Issue Resolver",
            code="P5",
            description="Time-sensitive, needs immediate information",
            characteristics=[
                "Emphasizes urgency",
                "Short, direct questions",
                "Deadline mentions",
                "Action-oriented language",
                "Stressed or pressed for time"
            ],
            example_phrases=[
                "I need this now",
                "Quick question",
                "Urgent",
                "Meeting in",
                "ASAP",
                "Time-sensitive"
            ]
        )
    }
    
    def __init__(self, bedrock_provider: Any):
        """
        Initialize PersonaTransformer.
        
        Args:
            bedrock_provider: BedrockProvider instance for LLM calls
        """
        self.provider = bedrock_provider
        logger.info(f"Initialized PersonaTransformer with {len(self.PERSONAS)} personas")
    
    def transform(
        self,
        query: str,
        personas: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Transform query into multiple persona variations.
        
        Args:
            query: Original query (Q)
            personas: List of persona keys to use (default: all 5)
            
        Returns:
            Dictionary mapping persona_code → transformed_query
            Example: {"P1": "...", "P2": "...", ...}
        """
        if personas is None:
            personas = list(self.PERSONAS.keys())
        
        logger.info(f"Transforming query into {len(personas)} personas")
        logger.debug(f"Original query: {query}")
        
        variations = {}
        
        for persona_key in personas:
            if persona_key not in self.PERSONAS:
                logger.warning(f"Unknown persona: {persona_key}, skipping")
                continue
            
            persona = self.PERSONAS[persona_key]
            
            try:
                transformed = self._transform_to_persona(query, persona)
                variations[persona.code] = transformed
                logger.debug(f"{persona.code} ({persona.name}): {transformed}")
                
            except Exception as e:
                logger.error(f"Failed to transform to {persona.name}: {e}")
                # Use original query as fallback
                variations[persona.code] = query
        
        logger.info(f"✅ Generated {len(variations)} persona variations")
        return variations
    
    def _transform_to_persona(
        self,
        query: str,
        persona: PersonaProfile
    ) -> str:
        """
        Transform query to match specific persona style.
        
        Uses LLM to rewrite query maintaining semantic intent.
        """
        prompt = f"""You are an expert at rewriting questions to match different communication styles.

ORIGINAL QUESTION:
{query}

PERSONA TO MATCH:
Name: {persona.name}
Description: {persona.description}

CHARACTERISTICS:
{chr(10).join(['- ' + c for c in persona.characteristics])}

EXAMPLE PHRASES:
{chr(10).join(['- "' + p + '"' for p in persona.example_phrases])}

TASK:
Rewrite the question to match this persona's communication style while preserving the EXACT same semantic intent and information need.

RULES:
1. Keep the core question and information need IDENTICAL
2. Change ONLY the communication style, tone, and phrasing
3. Maintain any specific details or constraints from the original
4. Do NOT add new information or change the question's scope
5. Output ONLY the rewritten question, no explanation

REWRITTEN QUESTION:"""
        
        transformed = self.provider.generate_text(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        return transformed.strip()
    
    def transform_batch(
        self,
        queries: List[str],
        personas: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Transform multiple queries into persona variations.
        
        Args:
            queries: List of original queries
            personas: Personas to use (default: all 5)
            
        Returns:
            List of dictionaries, one per query
        """
        logger.info(f"Batch transforming {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries, 1):
            try:
                variations = self.transform(query, personas)
                results.append(variations)
                logger.info(f"Completed {i}/{len(queries)}")
                
            except Exception as e:
                logger.error(f"Failed batch item {i}: {e}")
                results.append({})
        
        logger.info(f"✅ Batch complete: {len(results)} items processed")
        return results
    
    def get_persona_info(self, persona_key: str) -> Optional[PersonaProfile]:
        """Get information about a specific persona."""
        return self.PERSONAS.get(persona_key)
    
    def list_personas(self) -> List[str]:
        """List all available persona keys."""
        return list(self.PERSONAS.keys())
    
    def get_expansion_factor(self) -> int:
        """Get the expansion factor (number of personas)."""
        return len(self.PERSONAS)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PersonaTransformer(personas={len(self.PERSONAS)})"


if __name__ == "__main__":
    # Test the transformer
    import sys
    sys.path.append('/home/claude')
    
    from src.core.bedrock_provider import BedrockProvider
    from src.utils import setup_logger, load_config
    
    setup_logger("INFO")
    config = load_config()
    
    # Initialize
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        region=config.bedrock.region
    )
    
    transformer = PersonaTransformer(provider)
    
    print("\n" + "="*80)
    print("PERSONA TRANSFORMER TEST")
    print("="*80)
    
    # Test query
    test_query = "What is my current portfolio allocation?"
    
    print(f"\nOriginal Query: {test_query}\n")
    
    # Transform
    variations = transformer.transform(test_query)
    
    print("\nPERSONA VARIATIONS:")
    print("-"*80)
    for persona_code, transformed_query in variations.items():
        persona = next(p for p in transformer.PERSONAS.values() if p.code == persona_code)
        print(f"\n{persona_code} - {persona.name}:")
        print(f"  {transformed_query}")
    
    print("\n" + "="*80)
    print(f"✅ Generated {len(variations)} variations")
    print("="*80)