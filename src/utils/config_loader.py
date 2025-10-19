"""
Configuration loader and validator for Trajectory Synthetic Data Generator.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BedrockConfig:
    """AWS Bedrock configuration."""
    region: str
    model_id: str
    embedding_model_id: str
    max_tokens: int
    temperature: float


@dataclass
class PDFProcessingConfig:
    """PDF processing configuration."""
    extract_images: bool
    chunk_size: int
    chunk_overlap: int
    use_vision_for_images: bool


@dataclass
class ChromaDBConfig:
    """ChromaDB configuration."""
    persist_directory: str
    collection_name: str
    distance_metric: str


@dataclass
class ToolsConfig:
    """Tools configuration."""
    definitions_file: str
    enable_tool_use: bool = True


@dataclass
class GenerationConfig:
    """Generation settings configuration."""
    target_qa_pairs: int
    expansion_factor: int
    complexity_distribution: Dict[str, float]


@dataclass
class VariationsConfig:
    """Variation types configuration."""
    personas: list
    query_modifications: list
    tool_sequences: list


@dataclass
class CurationConfig:
    """Curation settings configuration."""
    similarity_threshold: float
    enable_dual_phase: bool
    min_quality_score: float


@dataclass
class OutputSchemaFields:
    """Output schema field names."""
    query: str = "Qi"
    cot: str = "COTi"
    tools: str = "Tool Set i"
    decision: str = "Decisioni"


@dataclass
class OutputSchema:
    """Output schema configuration."""
    type: str = "trajectory"
    fields: OutputSchemaFields = field(default_factory=OutputSchemaFields)
    include_reasoning: bool = True
    include_metadata: bool = True
    include_tool_results: bool = False


@dataclass
class OutputConfig:
    """Output settings configuration."""
    format: str
    output_dir: str
    save_trajectories: bool
    organize_by_complexity: bool
    schema: OutputSchema = field(default_factory=OutputSchema)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    file: str


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default path.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._raw_config = self._load_yaml()
        self._validate_config()
        self._parse_config()
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate required configuration sections exist."""
        required_sections = [
            'bedrock', 'pdf_processing', 'chromadb', 'generation',
            'variations', 'curation', 'output', 'logging'
        ]
        
        # Tools is optional for backward compatibility
        
        for section in required_sections:
            if section not in self._raw_config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _parse_config(self):
        """Parse configuration into dataclass objects."""
        # Bedrock configuration
        bedrock = self._raw_config['bedrock']
        self.bedrock = BedrockConfig(
            region=bedrock.get('region', 'us-east-1'),
            model_id=bedrock['model_id'],
            embedding_model_id=bedrock['embedding_model_id'],
            max_tokens=bedrock.get('max_tokens', 64000),
            temperature=bedrock.get('temperature', 0.7)
        )
        
        # PDF processing configuration
        pdf = self._raw_config['pdf_processing']
        self.pdf_processing = PDFProcessingConfig(
            extract_images=pdf.get('extract_images', True),
            chunk_size=pdf.get('chunk_size', 4000),
            chunk_overlap=pdf.get('chunk_overlap', 200),
            use_vision_for_images=pdf.get('use_vision_for_images', True)
        )
        
        # ChromaDB configuration
        chroma = self._raw_config['chromadb']
        self.chromadb = ChromaDBConfig(
            persist_directory=chroma['persist_directory'],
            collection_name=chroma.get('collection_name', 'document_chunks'),
            distance_metric=chroma.get('distance_metric', 'cosine')
        )
        
        # Tools configuration (optional)
        if 'tools' in self._raw_config:
            tools_cfg = self._raw_config['tools']
            self.tools = ToolsConfig(
                definitions_file=tools_cfg.get('definitions_file', 'config/tools.json'),
                enable_tool_use=tools_cfg.get('enable_tool_use', True)
            )
        else:
            # Default tools config
            self.tools = ToolsConfig(
                definitions_file='config/tools.json',
                enable_tool_use=True
            )
        
        # Generation configuration
        gen = self._raw_config['generation']
        self.generation = GenerationConfig(
            target_qa_pairs=gen.get('target_qa_pairs', 1500),
            expansion_factor=gen.get('expansion_factor', 90),
            complexity_distribution=gen.get('complexity_distribution', {
                'simple': 0.3,
                'medium': 0.5,
                'complex': 0.2
            })
        )
        
        # Variations configuration
        var = self._raw_config['variations']
        self.variations = VariationsConfig(
            personas=var.get('personas', []),
            query_modifications=var.get('query_modifications', []),
            tool_sequences=var.get('tool_sequences', [])
        )
        
        # Curation configuration
        cur = self._raw_config['curation']
        self.curation = CurationConfig(
            similarity_threshold=cur.get('similarity_threshold', 0.85),
            enable_dual_phase=cur.get('enable_dual_phase', True),
            min_quality_score=cur.get('min_quality_score', 7.0)
        )
        
        # Output configuration
        out = self._raw_config['output']
        
        # Parse output schema if present
        if 'schema' in out:
            schema_cfg = out['schema']
            
            # Parse fields
            if 'fields' in schema_cfg:
                fields_cfg = schema_cfg['fields']
                fields = OutputSchemaFields(
                    query=fields_cfg.get('query', 'Qi'),
                    cot=fields_cfg.get('cot', 'COTi'),
                    tools=fields_cfg.get('tools', 'Tool Set i'),
                    decision=fields_cfg.get('decision', 'Decisioni')
                )
            else:
                fields = OutputSchemaFields()
            
            schema = OutputSchema(
                type=schema_cfg.get('type', 'trajectory'),
                fields=fields,
                include_reasoning=schema_cfg.get('include_reasoning', True),
                include_metadata=schema_cfg.get('include_metadata', True),
                include_tool_results=schema_cfg.get('include_tool_results', False)
            )
        else:
            schema = OutputSchema()
        
        self.output = OutputConfig(
            format=out.get('format', 'jsonl'),
            output_dir=out.get('output_dir', 'data/output'),
            save_trajectories=out.get('save_trajectories', True),
            organize_by_complexity=out.get('organize_by_complexity', False),
            schema=schema
        )
        
        # Logging configuration
        log = self._raw_config['logging']
        self.logging = LoggingConfig(
            level=log.get('level', 'INFO'),
            file=log.get('file', 'logs/trajectory_generator.log')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._raw_config
    
    def __repr__(self) -> str:
        return f"Config(path='{self.config_path}')"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default path.
        
    Returns:
        Config object
    """
    return Config(config_path)