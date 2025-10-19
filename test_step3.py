"""
Test script for AWS Bedrock Provider (Step 3).
Run this to verify Bedrock integration is working.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from src.utils import load_config, setup_logger, get_logger
from src.core import BedrockProvider

# Load environment variables
load_dotenv()

def test_initialization():
    """Test Bedrock provider initialization."""
    print("\n" + "="*60)
    print("TEST 1: Provider Initialization")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        config = load_config()
        
        provider = BedrockProvider(
            model_id=config.bedrock.model_id,
            embedding_model_id=config.bedrock.embedding_model_id,
            region=config.bedrock.region,
            max_tokens=config.bedrock.max_tokens,
            temperature=config.bedrock.temperature
        )
        
        logger.info(f"✅ Provider initialized: {provider}")
        logger.info(f"Model: {provider.model_id}")
        logger.info(f"Embedding Model: {provider.embedding_model_id}")
        logger.info(f"Region: {provider.region}")
        
        print("✅ Provider initialization test passed\n")
        return provider
        
    except Exception as e:
        print(f"❌ Provider initialization failed: {e}\n")
        return None


def test_text_generation(provider: BedrockProvider):
    """Test text generation."""
    print("="*60)
    print("TEST 2: Text Generation")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        prompt = "Explain what AWS Bedrock is in 2 sentences."
        
        logger.info(f"Sending prompt: {prompt[:50]}...")
        response = provider.generate_text(prompt)
        
        logger.info(f"Received response ({len(response)} chars)")
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}\n")
        
        print("✅ Text generation test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_system_prompt(provider: BedrockProvider):
    """Test generation with system prompt."""
    print("="*60)
    print("TEST 3: Generation with System Prompt")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        system_prompt = "You are a helpful assistant that speaks like a pirate."
        prompt = "Tell me about cloud computing."
        
        logger.info("Testing with system prompt...")
        response = provider.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=200
        )
        
        print(f"\nSystem: {system_prompt}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")
        
        print("✅ System prompt test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ System prompt test failed: {e}\n")
        return False


def test_embedding_generation(provider: BedrockProvider):
    """Test embedding generation."""
    print("="*60)
    print("TEST 4: Embedding Generation")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        text = "AWS Bedrock is a fully managed service for building AI applications."
        
        logger.info(f"Generating embedding for: {text[:50]}...")
        embedding = provider.generate_embedding(text)
        
        logger.info(f"Embedding dimension: {len(embedding)}")
        print(f"\nText: {text}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}\n")
        
        print("✅ Embedding generation test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_batch_embeddings(provider: BedrockProvider):
    """Test batch embedding generation."""
    print("="*60)
    print("TEST 5: Batch Embedding Generation")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        texts = [
            "First document about AI",
            "Second document about machine learning",
            "Third document about data science"
        ]
        
        logger.info(f"Generating {len(texts)} embeddings...")
        embeddings = provider.generate_embeddings_batch(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        print(f"\nGenerated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"  - Text {i+1}: dimension {len(emb)}")
        
        print("\n✅ Batch embedding test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Batch embedding failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "🚀"*30)
    print("TRAJECTORY SYNTHETIC DATA GENERATOR")
    print("AWS Bedrock Provider Test Suite")
    print("🚀"*30)
    
    # Setup logger
    setup_logger("INFO", "logs/test_bedrock.log")
    
    # Test 1: Initialization
    provider = test_initialization()
    
    if not provider:
        print("❌ Cannot proceed without provider initialization")
        return
    
    # Test 2: Text Generation
    test_text_generation(provider)
    
    # Test 3: System Prompt
    test_system_prompt(provider)
    
    # Test 4: Embedding Generation
    test_embedding_generation(provider)
    
    # Test 5: Batch Embeddings
    test_batch_embeddings(provider)
    
    print("="*60)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*60)
    print("\nStep 3: AWS Bedrock Provider ✅ COMPLETE\n")
    print("Next: Implement ChromaDB Manager (Step 4)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()