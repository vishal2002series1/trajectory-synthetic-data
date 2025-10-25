"""
Comprehensive Test Suite for Multi-Iteration Trajectory Generation

Tests all components: DecisionEngine, IterationState, MultiIterGenerator

Location: tests/test_complete_suite.py

Run: python -m pytest tests/test_complete_suite.py -v
Or: python tests/test_complete_suite.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.core.bedrock_provider import BedrockProvider
from src.core.iteration_state import IterationState, StateManager, ToolResult
from src.generators.decision_engine import DecisionEngine, Decision, DecisionType
from src.generators.trajectory_generator_multi_iter import TrajectoryGeneratorMultiIter
from src.utils import load_config, setup_logger


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Load configuration."""
    return load_config()


@pytest.fixture
def bedrock_provider(config):
    """Create BedrockProvider instance."""
    return BedrockProvider(
        model_id=config.bedrock.model_id,
        region=config.bedrock.region
    )


@pytest.fixture
def mock_tools():
    """Mock tool definitions."""
    return [
        {
            "name": "search_knowledge_base",
            "description": "Search for relevant information"
        },
        {
            "name": "calculate",
            "description": "Perform calculations"
        }
    ]


# =============================================================================
# TEST 1: ITERATION STATE
# =============================================================================

class TestIterationState:
    """Test IterationState and StateManager."""
    
    def test_state_initialization(self):
        """Test state initialization."""
        manager = StateManager()
        state = manager.initialize("query_1", "What is my allocation?")
        
        assert state.query == "What is my allocation?"
        assert state.iteration == 0
        assert len(state.tool_results) == 0
    
    def test_add_tool_results(self):
        """Test adding tool results advances iteration."""
        manager = StateManager()
        state = manager.initialize("query_1", "What is my allocation?")
        
        # Add tool results
        result = ToolResult(
            tool_name="search_knowledge_base",
            parameters={"query": "allocation"},
            result={"data": "test"},
            iteration=0
        )
        state.add_tool_results([result])
        
        assert state.iteration == 1
        assert len(state.tool_results) == 1
    
    def test_context_formatting(self):
        """Test context formatting for LLM."""
        manager = StateManager()
        state = manager.initialize("query_1", "What is my allocation?")
        
        # Add tool results
        result = ToolResult(
            tool_name="search_knowledge_base",
            parameters={"query": "allocation"},
            result={"documents": ["doc1"]},
            iteration=0
        )
        state.add_tool_results([result])
        
        context = state.to_context()
        assert context["query"] == "What is my allocation?"
        assert context["iteration"] == 1
        assert len(context["tool_results"]) == 1


# =============================================================================
# TEST 2: DECISION ENGINE (Basic)
# =============================================================================

class TestDecisionEngineBasic:
    """Test DecisionEngine basic functionality."""
    
    def test_initialization(self, bedrock_provider):
        """Test decision engine initialization."""
        engine = DecisionEngine(bedrock_provider)
        assert engine.provider == bedrock_provider
    
    def test_decision_object_creation(self):
        """Test Decision object creation."""
        decision = Decision(
            type=DecisionType.CALL,
            reasoning="Need more data",
            tools=["search_knowledge_base"]
        )
        
        assert decision.type == DecisionType.CALL
        assert "search_knowledge_base" in decision.tools


# =============================================================================
# TEST 3: DECISION ENGINE (With LLM) - OPTIONAL
# =============================================================================

class TestDecisionEngineLLM:
    """Test DecisionEngine with actual LLM calls."""
    
    @pytest.mark.slow
    def test_iteration_0_decision(self, bedrock_provider, mock_tools):
        """Test decision at iteration 0."""
        engine = DecisionEngine(bedrock_provider)
        
        decision = engine.decide(
            query="What is my portfolio allocation?",
            context=[],
            available_tools=mock_tools,
            iteration=0,
            max_iterations=3
        )
        
        # At iteration 0, should typically CALL
        assert decision.type in [DecisionType.CALL, DecisionType.ASK]
        assert decision.reasoning is not None
    
    @pytest.mark.slow
    def test_iteration_with_context(self, bedrock_provider, mock_tools):
        """Test decision with context."""
        engine = DecisionEngine(bedrock_provider)
        
        # Simulate having data
        context = [
            {
                "tool": "search_knowledge_base",
                "data": {"allocation": {"stocks": 60, "bonds": 40}}
            }
        ]
        
        decision = engine.decide(
            query="What is my portfolio allocation?",
            context=context,
            available_tools=mock_tools,
            iteration=1,
            max_iterations=3
        )
        
        # With context, should typically ANSWER
        # But might also CALL for more info
        assert decision.type in [DecisionType.ANSWER, DecisionType.CALL]


# =============================================================================
# TEST 4: MULTI-ITERATION GENERATOR
# =============================================================================

class TestMultiIterationGenerator:
    """Test TrajectoryGeneratorMultiIter."""
    
    def test_initialization(self, bedrock_provider, config):
        """Test generator initialization."""
        generator = TrajectoryGeneratorMultiIter(
            bedrock_provider=bedrock_provider,
            config=config,
            max_iterations=3,
            use_mock_tools=True
        )
        
        assert generator.max_iterations == 3
        assert generator.use_mock_tools is True
    
    def test_mock_tool_execution(self, bedrock_provider, config):
        """Test mock tool execution."""
        generator = TrajectoryGeneratorMultiIter(
            bedrock_provider=bedrock_provider,
            config=config,
            use_mock_tools=True
        )
        
        result = generator._mock_tool_execution("search_knowledge_base", "test query")
        
        assert "documents" in result
        assert len(result["documents"]) > 0
    
    @pytest.mark.slow
    def test_generate_trajectory(self, bedrock_provider, config):
        """Test full trajectory generation."""
        generator = TrajectoryGeneratorMultiIter(
            bedrock_provider=bedrock_provider,
            config=config,
            max_iterations=3,
            use_mock_tools=True
        )
        
        examples = generator.generate_trajectory(
            query="What is my current portfolio allocation?"
        )
        
        # Should generate at least 1 example, likely 2-3
        assert len(examples) >= 1
        assert len(examples) <= 3
        
        # Check first example format
        first_ex = examples[0]
        assert first_ex.query == "What is my current portfolio allocation?"
        assert first_ex.chain_of_thought is not None
        assert first_ex.metadata["iteration"] == 0


# =============================================================================
# TEST 5: TRAINING EXAMPLE FORMAT
# =============================================================================

class TestTrainingExampleFormat:
    """Test training example output format."""
    
    @pytest.mark.slow
    def test_output_format(self, bedrock_provider, config):
        """Test training example matches config format."""
        generator = TrajectoryGeneratorMultiIter(
            bedrock_provider=bedrock_provider,
            config=config,
            max_iterations=2,
            use_mock_tools=True
        )
        
        examples = generator.generate_trajectory(
            query="How has my portfolio performed this year?"
        )
        
        # Convert to dict format
        example_dict = examples[0].to_dict(generator.field_names)
        
        # Check field names match config
        assert config.output.schema.fields.query in example_dict
        assert config.output.schema.fields.cot in example_dict
        assert config.output.schema.fields.tools in example_dict
        assert config.output.schema.fields.decision in example_dict


# =============================================================================
# MANUAL TESTS (Run without pytest)
# =============================================================================

def manual_test_full_pipeline():
    """
    Manual test of full pipeline.
    Run: python tests/test_complete_suite.py
    """
    print("\n" + "="*80)
    print("MANUAL TEST: FULL PIPELINE")
    print("="*80)
    
    setup_logger("INFO")
    config = load_config()
    
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        region=config.bedrock.region
    )
    
    generator = TrajectoryGeneratorMultiIter(
        bedrock_provider=provider,
        config=config,
        max_iterations=3,
        use_mock_tools=True
    )
    
    # Test 1: Simple query
    print("\n" + "-"*80)
    print("TEST 1: Simple allocation query")
    print("-"*80)
    
    examples_1 = generator.generate_trajectory(
        query="What is my current portfolio allocation?"
    )
    
    print(f"✅ Generated {len(examples_1)} training examples")
    for i, ex in enumerate(examples_1, 1):
        print(f"\nExample {i}:")
        print(f"  Iteration: {ex.metadata['iteration']}")
        print(f"  Decision Type: {ex.metadata['decision_type']}")
        print(f"  Has Context: {ex.context is not None}")
        print(f"  Tools Called: {len(ex.tool_set)}")
    
    # Test 2: Complex query
    print("\n" + "-"*80)
    print("TEST 2: Complex performance query")
    print("-"*80)
    
    examples_2 = generator.generate_trajectory(
        query="How has my portfolio performed this year compared to my target return?"
    )
    
    print(f"✅ Generated {len(examples_2)} training examples")
    
    # Test 3: Save examples
    print("\n" + "-"*80)
    print("TEST 3: Save training examples")
    print("-"*80)
    
    all_examples = examples_1 + examples_2
    output_file = Path("data/output/manual_test_examples.jsonl")
    generator.save_training_examples(all_examples, output_file, format="jsonl")
    
    print(f"✅ Saved {len(all_examples)} examples to {output_file}")
    
    # Show sample output
    print("\n" + "-"*80)
    print("SAMPLE OUTPUT (First Example)")
    print("-"*80)
    
    sample = examples_1[0].to_dict(generator.field_names)
    print(json.dumps(sample, indent=2))
    
    print("\n" + "="*80)
    print("✅ ALL MANUAL TESTS PASSED")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run manual tests if executed directly
    manual_test_full_pipeline()
else:
    # Run pytest tests if imported
    pytest.main([__file__, "-v", "-m", "not slow"])