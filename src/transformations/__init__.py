"""
Transformation modules for synthetic data generation.

These transformations expand seed data by applying variations:
- Persona Variation: 5 communication styles
- Query Modification: 3 complexity levels
- Tool Data Variation: Correct/Wrong data
- PDF Augmentation: Context addition & new generation
- Multi-turn Expansion: Conversation sequences
"""

from .persona_transformer import PersonaTransformer
from .query_modifier import QueryModifier
from .tool_data_transformer import ToolDataTransformer
from .context_variation_transformer import ContextVariationTransformer

__all__ = [
    'PersonaTransformer',
    'QueryModifier',
    'ToolDataTransformer',
    'ContextVariationTransformer',
]