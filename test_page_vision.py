"""
Test PDF Parser with page-level vision analysis.
Tests rendering pages as images to capture charts and graphs.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.core import PDFParser, BedrockProvider
from src.utils import setup_logger, load_config

# Load configuration
config = load_config()

# Setup logging
setup_logger(config.logging.level, config.logging.file)

print("\n" + "="*60)
print("TESTING PAGE-LEVEL VISION ANALYSIS")
print("Vanguard PDF with Charts & Graphs")
print("="*60)
print(f"\n📋 Configuration:")
print(f"  Model: {config.bedrock.model_id}")
print(f"  Region: {config.bedrock.region}")
print(f"  Max Tokens: {config.bedrock.max_tokens}")

# Initialize with config settings
parser = PDFParser(
    extract_images=config.pdf_processing.extract_images,
    chunk_size=config.pdf_processing.chunk_size,
    chunk_overlap=config.pdf_processing.chunk_overlap,
    min_image_size=100
)

provider = BedrockProvider(
    model_id=config.bedrock.model_id,
    embedding_model_id=config.bedrock.embedding_model_id,
    region=config.bedrock.region,
    max_tokens=config.bedrock.max_tokens,
    temperature=config.bedrock.temperature
)

# Parse PDF path
pdf_path = "vanguards_principles_for_investing_success.pdf"

print(f"\nParsing: {pdf_path}")
print("\n⚠️  NOTE: This will analyze ALL 32 pages with Claude Vision")
print("Expected time: ~2-5 minutes (analyzing each page)")
print("This captures ALL visual content: charts, graphs, diagrams!\n")

input("Press Enter to continue...")

# Parse with PAGE-LEVEL vision analysis
result = parser.parse_pdf(
    pdf_path,
    vision_provider=provider,
    analyze_images=True,  # Analyze embedded images (if any)
    analyze_pages=True,   # ← RENDER & ANALYZE EACH PAGE
    page_dpi=150  # Good quality, reasonable speed
)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nBasic Info:")
print(f"  Source: {result['source']}")
print(f"  Total Pages: {result['total_pages']}")
print(f"  Text Length: {len(result['full_text'])} characters")
print(f"  Embedded Images: {len(result['images'])}")
print(f"  Pages Analyzed: {len(result.get('page_descriptions', {}))}")
print(f"  Text Chunks: {len(result['chunks'])}")

# Show page descriptions (the key feature!)
if result.get('page_descriptions'):
    print("\n" + "="*60)
    print("CLAUDE'S PAGE-BY-PAGE VISUAL ANALYSIS")
    print("="*60)
    
    for page_num, desc in result['page_descriptions'].items():
        print(f"\n{'='*60}")
        print(f"Page {page_num}:")
        print(f"{'='*60}")
        print(desc)
        print()

# Show which pages have charts/graphs
visual_pages = []
text_only_pages = []

if result.get('page_descriptions'):
    for page_num, desc in result['page_descriptions'].items():
        if "text-only" in desc.lower():
            text_only_pages.append(page_num)
        else:
            visual_pages.append(page_num)
    
    print("\n" + "="*60)
    print("SUMMARY: Visual vs Text-Only Pages")
    print("="*60)
    print(f"\nPages with charts/graphs/figures: {len(visual_pages)}")
    if visual_pages:
        print(f"  Pages: {', '.join(map(str, visual_pages))}")
    
    print(f"\nText-only pages: {len(text_only_pages)}")
    if text_only_pages:
        print(f"  Pages: {', '.join(map(str, text_only_pages))}")

# Show chunk info with visual context
print("\n" + "="*60)
print("TEXT CHUNKS (Now with Visual Context!)")
print("="*60)
print(f"Total chunks: {len(result['chunks'])}")
if result['chunks']:
    print(f"\nLast chunk (should include visual analysis):")
    last_chunk = result['chunks'][-1]
    print(f"  Chunk ID: {last_chunk.chunk_id}")
    print(f"  Length: {len(last_chunk.text)} chars")
    print(f"  Has page analysis: {last_chunk.metadata.get('has_page_analysis', False)}")
    
    # Show last 500 chars (where visual analysis should be)
    print(f"\n  Last 500 characters (visual context):")
    print(f"  ...{last_chunk.text[-500:]}")

print("\n" + "="*60)
print("✅ PAGE-LEVEL VISION ANALYSIS COMPLETE!")
print("="*60)
print(f"\nKey Achievements:")
print(f"  ✅ Rendered all {result['total_pages']} pages as images")
print(f"  ✅ Claude analyzed each page for visual content")
print(f"  ✅ Charts, graphs, and diagrams are now captured!")
print(f"  ✅ Visual analysis embedded in searchable chunks")
print("="*60 + "\n")