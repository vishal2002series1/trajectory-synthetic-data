"""
Test Vision/Multi-modal capabilities of BedrockProvider.
Tests the generate_with_vision() method with images.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from PIL import Image, ImageDraw, ImageFont
import io
from src.utils import load_config, setup_logger, get_logger
from src.core import BedrockProvider

def create_test_image():
    """Create a simple test image with text."""
    print("\n" + "="*60)
    print("Creating Test Image")
    print("="*60)
    
    # Create a 400x300 image with white background
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes and text
    # Rectangle
    draw.rectangle([50, 50, 200, 150], fill='red', outline='black', width=3)
    
    # Circle
    draw.ellipse([220, 50, 350, 180], fill='blue', outline='black', width=3)
    
    # Text
    draw.text((150, 220), "Test Image", fill='black')
    draw.text((120, 250), "Red Square + Blue Circle", fill='black')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Also save to file for reference
    img.save('data/output/test_vision_image.png')
    print("✅ Created test image: data/output/test_vision_image.png")
    
    return img_bytes.getvalue()


def test_vision_from_bytes(provider: BedrockProvider):
    """Test vision with image bytes."""
    print("\n" + "="*60)
    print("TEST 1: Vision from Bytes")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # Create test image
        image_bytes = create_test_image()
        
        # Test vision
        prompt = "Describe what you see in this image. Be specific about colors and shapes."
        
        logger.info("Sending image with prompt...")
        response = provider.generate_with_vision(
            prompt=prompt,
            image_bytes=image_bytes,
            image_media_type="image/png"
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}\n")
        
        print("✅ Vision from bytes test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Vision from bytes failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_vision_from_file(provider: BedrockProvider):
    """Test vision with image file path."""
    print("="*60)
    print("TEST 2: Vision from File Path")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # Use the image we created
        image_path = 'data/output/test_vision_image.png'
        
        prompt = "What shapes and colors do you see? Count them."
        
        logger.info(f"Loading image from: {image_path}")
        response = provider.generate_with_vision(
            prompt=prompt,
            image_path=image_path
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}\n")
        
        print("✅ Vision from file test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Vision from file failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_vision_with_analysis(provider: BedrockProvider):
    """Test vision with detailed analysis prompt."""
    print("="*60)
    print("TEST 3: Detailed Image Analysis")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        image_path = 'data/output/test_vision_image.png'
        
        prompt = """Analyze this image and provide:
1. List of all shapes you see
2. Colors of each shape
3. Any text present
4. Overall layout description"""
        
        logger.info("Requesting detailed analysis...")
        response = provider.generate_with_vision(
            prompt=prompt,
            image_path=image_path,
            max_tokens=500
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}\n")
        
        print("✅ Detailed analysis test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_vision_with_system_prompt(provider: BedrockProvider):
    """Test vision with system prompt."""
    print("="*60)
    print("TEST 4: Vision with System Prompt")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        image_path = 'data/output/test_vision_image.png'
        
        system_prompt = "You are an art critic. Analyze images with sophistication and detail."
        prompt = "Critique this image as if it were a work of modern art."
        
        logger.info("Testing with system prompt...")
        response = provider.generate_with_vision(
            prompt=prompt,
            image_path=image_path,
            system_prompt=system_prompt,
            max_tokens=300
        )
        
        print(f"\nSystem: {system_prompt}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")
        
        print("✅ System prompt with vision test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ System prompt with vision failed: {e}\n")
        return False


def main():
    """Run all vision tests."""
    print("\n" + "👁️"*30)
    print("VISION/MULTI-MODAL TEST SUITE")
    print("👁️"*30)
    
    # Setup logger
    setup_logger("INFO", "logs/test_vision.log")
    
    # Initialize provider
    config = load_config()
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        embedding_model_id=config.bedrock.embedding_model_id,
        region=config.bedrock.region,
        max_tokens=config.bedrock.max_tokens,
        temperature=config.bedrock.temperature
    )
    
    print(f"\n✅ Provider initialized: {provider}\n")
    
    # Run tests
    results = []
    
    results.append(("Vision from Bytes", test_vision_from_bytes(provider)))
    results.append(("Vision from File", test_vision_from_file(provider)))
    results.append(("Detailed Analysis", test_vision_with_analysis(provider)))
    results.append(("System Prompt + Vision", test_vision_with_system_prompt(provider)))
    
    # Summary
    print("="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL VISION TESTS PASSED!")
        print("="*60)
        print("\nStep 3: Multi-modal Support ✅ VERIFIED\n")
        print("Now proceeding to Step 4: ChromaDB Manager")
    else:
        print("⚠️  SOME VISION TESTS FAILED")
        print("="*60)
        print("\nVision support may have issues.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()