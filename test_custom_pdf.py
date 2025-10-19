# Quick test script
from src.core import PDFParser, BedrockProvider

parser = PDFParser(extract_images=True, min_image_size=100)
provider = BedrockProvider()

result = parser.parse_pdf(
    "Kia-K2700-2019-ZA.pdf",
    vision_provider=provider,
    analyze_images=True
)

print(f"Pages: {result['total_pages']}")
print(f"Text: {len(result['full_text'])} chars")
print(f"Images: {len(result['images'])}")
print(f"Chunks: {len(result['chunks'])}")

# See image descriptions
for idx, desc in result['image_descriptions'].items():
    print(f"\nImage {idx+1}: {desc[:200]}...")