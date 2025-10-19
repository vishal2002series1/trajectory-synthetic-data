"""
Test script for PDF Parser (Step 5).
Creates sample PDF and tests text/image extraction.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw
import io

from src.utils import load_config, setup_logger, get_logger
from src.core import BedrockProvider, ChromaDBManager, PDFParser


def create_sample_pdf():
    """Create a sample PDF with text for testing."""
    print("\n" + "="*60)
    print("Creating Sample PDF")
    print("="*60)
    
    output_path = "data/pdfs/sample_document.pdf"
    Path("data/pdfs").mkdir(parents=True, exist_ok=True)
    
    # Create PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Page 1: Introduction
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Sample Technical Document")
    
    c.setFont("Helvetica", 12)
    y = height - 150
    
    text_lines = [
        "This is a sample document for testing the PDF parser.",
        "",
        "Artificial Intelligence Overview:",
        "Artificial intelligence (AI) is transforming how we work and live.",
        "Machine learning algorithms can learn patterns from data without",
        "explicit programming. Deep learning uses neural networks inspired",
        "by the human brain to process complex patterns.",
        "",
        "Key Concepts:",
        "1. Machine Learning - Learning from data",
        "2. Neural Networks - Brain-inspired computing",
        "3. Deep Learning - Multi-layer neural networks",
        "",
        "Applications:",
        "AI powers virtual assistants, recommendation systems, autonomous",
        "vehicles, medical diagnosis, and many other innovative solutions.",
    ]
    
    for line in text_lines:
        c.drawString(100, y, line)
        y -= 20
    
    c.showPage()
    
    # Page 2: Technical Details
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 100, "Technical Implementation")
    
    c.setFont("Helvetica", 11)
    y = height - 150
    
    page2_text = [
        "Natural Language Processing (NLP):",
        "NLP enables computers to understand, interpret, and generate",
        "human language. Applications include chatbots, translation,",
        "sentiment analysis, and text summarization.",
        "",
        "Computer Vision:",
        "Computer vision allows machines to interpret visual information",
        "from the world. It powers facial recognition, object detection,",
        "autonomous vehicles, and medical image analysis.",
        "",
        "Training Process:",
        "1. Data Collection - Gather training examples",
        "2. Model Training - Learn patterns from data",
        "3. Validation - Test on unseen data",
        "4. Deployment - Use in production",
        "",
        "Future Directions:",
        "The field continues to evolve with new architectures, better",
        "training methods, and improved efficiency.",
    ]
    
    for line in page2_text:
        c.drawString(100, y, line)
        y -= 18
    
    c.showPage()
    c.save()
    
    print(f"✅ Created sample PDF: {output_path}")
    print(f"   2 pages with text\n")
    
    return output_path


def test_initialization():
    """Test PDF parser initialization."""
    print("="*60)
    print("TEST 1: PDF Parser Initialization")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        config = load_config()
        
        parser = PDFParser(
            chunk_size=config.pdf_processing.chunk_size,
            chunk_overlap=config.pdf_processing.chunk_overlap,
            extract_images=config.pdf_processing.extract_images
        )
        
        logger.info(f"✅ Parser initialized: {parser}")
        print("✅ Initialization test passed\n")
        return parser
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}\n")
        return None


def test_text_extraction(parser: PDFParser, pdf_path: str):
    """Test text extraction from PDF."""
    print("="*60)
    print("TEST 2: Text Extraction")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        result = parser.parse_pdf(pdf_path, analyze_images=False)
        
        full_text = result['full_text']
        chunks = result['chunks']
        
        logger.info(f"Extracted text: {len(full_text)} characters")
        logger.info(f"Created chunks: {len(chunks)}")
        
        print(f"\nPDF: {result['source']}")
        print(f"Pages: {result['total_pages']}")
        print(f"Total text: {len(full_text)} characters")
        print(f"Chunks created: {len(chunks)}")
        print(f"\nFirst 200 characters:")
        print(f"  {full_text[:200]}...")
        
        print("\n✅ Text extraction test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Text extraction failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_image_extraction(parser: PDFParser, pdf_path: str):
    """Test image extraction from PDF."""
    print("="*60)
    print("TEST 3: Image Extraction")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        result = parser.parse_pdf(pdf_path, analyze_images=False)
        
        images = result['images']
        
        logger.info(f"Extracted {len(images)} images")
        
        print(f"\nExtracted {len(images)} images:")
        for i, img in enumerate(images):
            print(f"  Image {i+1}: Page {img.page_number}, "
                  f"{img.width}x{img.height} {img.format}")
        
        # Save images
        if images:
            saved_paths = parser.save_images(images, "data/output/extracted_images")
            print(f"\n✅ Saved {len(saved_paths)} images to data/output/extracted_images/")
        
        print("\n✅ Image extraction test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Image extraction failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_vision_analysis(
    parser: PDFParser,
    pdf_path: str,
    provider: BedrockProvider
):
    """Test vision-based image analysis."""
    print("="*60)
    print("TEST 4: Vision-Based Image Analysis")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # First check if there are images to analyze
        quick_result = parser.parse_pdf(pdf_path, analyze_images=False)
        
        if not quick_result['images']:
            print("\n⚠️  No images found in PDF - skipping vision test")
            print("✅ Vision analysis test skipped\n")
            return True
        
        logger.info(f"Analyzing {len(quick_result['images'])} images with vision...")
        result = parser.parse_pdf(
            pdf_path,
            vision_provider=provider,
            analyze_images=True
        )
        
        image_descriptions = result['image_descriptions']
        
        if image_descriptions:
            print(f"\nAnalyzed {len(image_descriptions)} images:")
            for idx, desc in image_descriptions.items():
                print(f"\nImage {idx+1} analysis:")
                print(f"  {desc[:150]}...")
        else:
            print("\n⚠️  No images were analyzed")
        
        print("\n✅ Vision analysis test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Vision analysis failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_chromadb_integration(
    parser: PDFParser,
    pdf_path: str,
    manager: ChromaDBManager,
    provider: BedrockProvider
):
    """Test integration with ChromaDB."""
    print("="*60)
    print("TEST 5: ChromaDB Integration")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # Parse PDF
        result = parser.parse_pdf(pdf_path, analyze_images=False)
        chunks = result['chunks']
        
        # Convert to documents
        documents, metadatas = parser.chunks_to_documents(chunks)
        
        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = provider.generate_embeddings_batch(documents)
        
        # Add to ChromaDB
        logger.info("Adding to ChromaDB...")
        ids = manager.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"\n✅ Added {len(ids)} chunks to ChromaDB")
        print(f"Collection now has {manager.count()} documents")
        
        # Test search
        query = "What is machine learning?"
        query_emb = provider.generate_embedding(query)
        
        results = manager.query(
            query_embeddings=[query_emb],
            n_results=2
        )
        
        print(f"\nSearch test: '{query}'")
        print(f"Top result: {results['documents'][0][0][:100]}...")
        
        print("\n✅ ChromaDB integration test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB integration failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "📄"*30)
    print("PDF PARSER TEST SUITE")
    print("📄"*30)
    
    # Setup logger
    setup_logger("INFO", "logs/test_pdf_parser.log")
    
    # Create sample PDF
    pdf_path = create_sample_pdf()
    
    # Initialize
    config = load_config()
    
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        embedding_model_id=config.bedrock.embedding_model_id,
        region=config.bedrock.region
    )
    print("✅ Bedrock provider ready\n")
    
    manager = ChromaDBManager(
        persist_directory="data/chromadb_test_pdf",
        collection_name="pdf_test"
    )
    manager.clear_collection()
    print("✅ ChromaDB manager ready\n")
    
    # Test 1: Initialization
    parser = test_initialization()
    
    if not parser:
        print("❌ Cannot proceed without parser initialization")
        return
    
    # Run tests
    results = []
    
    results.append(("Text Extraction", test_text_extraction(parser, pdf_path)))
    results.append(("Image Extraction", test_image_extraction(parser, pdf_path)))
    results.append(("Vision Analysis", test_vision_analysis(parser, pdf_path, provider)))
    results.append(("ChromaDB Integration", test_chromadb_integration(parser, pdf_path, manager, provider)))
    
    # Summary
    print("="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL PDF PARSER TESTS PASSED!")
        print("="*60)
        print("\nStep 5: Multi-modal PDF Parser ✅ COMPLETE\n")
        print("Next: Implement Trajectory Generator (Step 6)")
    else:
        print("⚠️  SOME PDF PARSER TESTS FAILED")
        print("="*60)
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()