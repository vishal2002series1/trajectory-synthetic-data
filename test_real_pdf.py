"""
Test PDF Parser with real PDF containing images and text.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.core import PDFParser, BedrockProvider
from src.utils import setup_logger

# Setup logging
setup_logger("INFO", "logs/test_real_pdf.log")

print("\n" + "="*60)
print("TESTING WITH REAL PDF: Kia-K2700-2019-ZA.pdf")
print("="*60)

# Initialize
parser = PDFParser(extract_images=True, min_image_size=100)
provider = BedrockProvider()

# Parse PDF path
pdf_path = "/mnt/user-data/uploads/Kia-K2700-2019-ZA.pdf"
pdf_path="vanguards_principles_for_investing_success.pdf"

print(f"\nParsing: {pdf_path}")
print("This may take a few minutes for image analysis...\n")

# Parse with vision analysis
result = parser.parse_pdf(
    pdf_path,
    vision_provider=provider,
    analyze_images=True  # Analyze images with Claude Vision
)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nBasic Info:")
print(f"  Source: {result['source']}")
print(f"  Total Pages: {result['total_pages']}")
print(f"  Text Length: {len(result['full_text'])} characters")
print(f"  Images Extracted: {len(result['images'])}")
print(f"  Text Chunks: {len(result['chunks'])}")

# Show first 500 chars of text
print(f"\n" + "-"*60)
print("First 500 characters of text:")
print("-"*60)
print(result['full_text'])
print("...\n")

# Show image details
if result['images']:
    print("-"*60)
    print(f"Extracted {len(result['images'])} Images:")
    print("-"*60)
    for i, img in enumerate(result['images']):
        print(f"\nImage {i+1}:")
        print(f"  Page: {img.page_number}")
        print(f"  Size: {img.width}x{img.height} pixels")
        print(f"  Format: {img.format}")

# Show image descriptions from vision analysis
if result['image_descriptions']:
    print("\n" + "="*60)
    print("CLAUDE VISION ANALYSIS")
    print("="*60)
    for idx, desc in result['image_descriptions'].items():
        print(f"\n{'='*60}")
        print(f"Image {idx+1} Analysis:")
        print(f"{'='*60}")
        print(desc)

# Show chunk info
print("\n" + "="*60)
print("TEXT CHUNKS")
print("="*60)
print(f"Total chunks: {len(result['chunks'])}")
if result['chunks']:
    print(f"\nFirst chunk preview:")
    print(f"  Chunk ID: {result['chunks'][0].chunk_id}")
    print(f"  Page: {result['chunks'][0].page_number}")
    print(f"  Length: {len(result['chunks'][0].text)} chars")
    print(f"  Text preview: {result['chunks'][0].text[:200]}...")

# Save images to disk
if result['images']:
    print("\n" + "="*60)
    print("SAVING IMAGES")
    print("="*60)
    saved_paths = parser.save_images(
        result['images'],
        "data/output/kia_pdf_images"
    )
    print(f"Saved {len(saved_paths)} images to: data/output/kia_pdf_images/")
    for path in saved_paths:
        print(f"  - {path}")

print("\n" + "="*60)
print("✅ PDF PARSING COMPLETE!")
print("="*60)
print(f"\nSummary:")
print(f"  - Extracted text from {result['total_pages']} pages")
print(f"  - Found {len(result['images'])} images")
print(f"  - Analyzed {len(result['image_descriptions'])} images with Claude Vision")
print(f"  - Created {len(result['chunks'])} searchable chunks")
print("="*60 + "\n")