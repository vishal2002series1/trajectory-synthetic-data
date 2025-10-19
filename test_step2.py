"""
Test script for core utilities.
Run this to verify Step 2 is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from src.utils import load_config, setup_logger, get_logger, write_json, read_json

# Load environment variables
load_dotenv()

def test_logger():
    """Test logger setup."""
    print("\n" + "="*60)
    print("TEST 1: Logger Setup")
    print("="*60)
    
    logger = setup_logger("DEBUG", "logs/test.log")
    logger.info("✅ Logger initialized successfully")
    logger.debug("Debug message test")
    logger.success("Success message test")
    logger.warning("Warning message test")
    
    print("✅ Logger test passed\n")


def test_config_loader():
    """Test configuration loader."""
    print("="*60)
    print("TEST 2: Configuration Loader")
    print("="*60)
    
    try:
        config = load_config()
        logger = get_logger(__name__)
        
        logger.info(f"Configuration loaded: {config}")
        logger.info(f"Model ID: {config.bedrock.model_id}")
        logger.info(f"Embedding Model: {config.bedrock.embedding_model_id}")
        logger.info(f"Region: {config.bedrock.region}")
        logger.info(f"Max Tokens: {config.bedrock.max_tokens}")
        logger.info(f"Chunk Size: {config.pdf_processing.chunk_size}")
        logger.info(f"ChromaDB Collection: {config.chromadb.collection_name}")
        logger.info(f"Target QA Pairs: {config.generation.target_qa_pairs}")
        logger.info(f"Expansion Factor: {config.generation.expansion_factor}")
        logger.info(f"Output Format: {config.output.format}")
        
        print("✅ Configuration loader test passed\n")
        return config
    except Exception as e:
        print(f"❌ Configuration loader test failed: {e}\n")
        return None


def test_file_utils():
    """Test file utilities."""
    print("="*60)
    print("TEST 3: File Utilities")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # Test JSON operations
        test_data = {
            "test": "data",
            "number": 123,
            "nested": {"key": "value"}
        }
        
        test_file = "data/output/test_file.json"
        write_json(test_data, test_file)
        logger.info(f"Wrote test JSON file: {test_file}")
        
        read_data = read_json(test_file)
        assert read_data == test_data, "JSON data mismatch"
        logger.info("Read and verified JSON file")
        
        print("✅ File utilities test passed\n")
    except Exception as e:
        print(f"❌ File utilities test failed: {e}\n")


def main():
    """Run all tests."""
    print("\n" + "🚀"*30)
    print("TRAJECTORY SYNTHETIC DATA GENERATOR")
    print("Core Utilities Test Suite")
    print("🚀"*30)
    
    # Test 1: Logger
    test_logger()
    
    # Test 2: Config Loader
    config = test_config_loader()
    
    # Test 3: File Utils
    if config:
        test_file_utils()
    
    print("="*60)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*60)
    print("\nStep 2: Core Utilities ✅ COMPLETE\n")
    print("Next: Implement AWS Bedrock Provider (Step 3)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()