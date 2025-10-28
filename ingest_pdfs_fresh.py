"""
Fresh PDF Ingestion - Using Existing Components Only

This script uses your existing components to re-ingest PDFs:
- VectorStore (from src.core)
- PDFParser (from src.core)
- load_config (from src.utils)

Run this after updating collection_name in config.yaml
"""

from pathlib import Path
from src.core import PDFParser, VectorStore
from src.utils import load_config, setup_logger

print("\n" + "="*80)
print("PDF INGESTION - USING EXISTING COMPONENTS")
print("="*80)

# Load config (will use new collection name)
config = load_config()
setup_logger("INFO")

print(f"\n📋 Configuration:")
print(f"   Collection: {config.chromadb.collection_name}")
print(f"   Persist Dir: {config.chromadb.persist_directory}")
print(f"   Chunk Size: {config.pdf_processing.chunk_size}")

# Initialize VectorStore (your existing component)
print(f"\n📊 Initializing VectorStore...")
vector_store = VectorStore(config=config)
current_count = vector_store.count()
print(f"✅ VectorStore initialized")
print(f"   Current documents: {current_count}")

# Check for PDFs
pdf_dir = Path("data/pdfs")
if not pdf_dir.exists():
    print(f"\n❌ Directory not found: {pdf_dir}")
    print("   Create it and add PDF files")
    exit(1)

pdf_files = sorted(pdf_dir.glob("*.pdf"))

if not pdf_files:
    print(f"\n❌ No PDF files found in {pdf_dir}")
    print("   Add PDF files and run again")
    exit(1)

print(f"\n📄 Found {len(pdf_files)} PDF files:")
for i, pdf in enumerate(pdf_files, 1):
    print(f"   {i}. {pdf.name}")

# Confirm
print(f"\n⚠️  This will add {len(pdf_files)} PDFs to collection '{config.chromadb.collection_name}'")
confirm = input("Proceed? [y/N]: ")

if confirm.lower() != 'y':
    print("❌ Cancelled")
    exit(0)

# Initialize PDF Parser (your existing component)
print(f"\n📄 Initializing PDF Parser...")
parser = PDFParser(
    chunk_size=config.pdf_processing.chunk_size,
    chunk_overlap=config.pdf_processing.chunk_overlap,
    extract_images=config.pdf_processing.extract_images
)
print("✅ PDF Parser initialized")

# Process each PDF
print("\n" + "="*80)
print("PROCESSING PDFs")
print("="*80)

total_chunks = 0
successful = 0
failed = 0

for i, pdf_file in enumerate(pdf_files, 1):
    print(f"\n[{i}/{len(pdf_files)}] {pdf_file.name}")
    
    try:
        # Parse PDF (existing method)
        print(f"   ⏳ Parsing...")
        result = parser.parse_pdf(str(pdf_file), analyze_images=False)
        chunks = result['chunks']
        
        print(f"   📝 Extracted {len(chunks)} chunks from {result['total_pages']} pages")
        
        # Add to vector store (existing method)
        print(f"   ⏳ Adding to vector store...")
        vector_store.add_chunks(chunks, source=pdf_file.name)
        
        total_chunks += len(chunks)
        successful += 1
        print(f"   ✅ Success ({len(chunks)} chunks added)")
        
    except Exception as e:
        failed += 1
        print(f"   ❌ Error: {e}")
        continue

# Final summary
print("\n" + "="*80)
print("INGESTION COMPLETE")
print("="*80)

final_count = vector_store.count()

print(f"\n📊 Summary:")
print(f"   PDFs processed: {successful}/{len(pdf_files)}")
print(f"   Failed: {failed}")
print(f"   Total chunks: {total_chunks}")
print(f"   Collection size: {current_count} → {final_count} (+{final_count - current_count})")

# Test query
print(f"\n🧪 Testing Query...")
try:
    test_results = vector_store.query("portfolio allocation", n_results=3)
    
    if test_results and test_results['documents'][0]:
        print(f"✅ Query successful!")
        print(f"   Retrieved {len(test_results['documents'][0])} results")
        print(f"\n   Sample result:")
        print(f"   \"{test_results['documents'][0][0][:150]}...\"")
    else:
        print(f"⚠️  Query returned no results")
        
except Exception as e:
    print(f"❌ Query failed: {e}")
    print(f"\n⚠️  There may still be an issue with embeddings")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Verify query works:")
print("   python -c \"from src.core import VectorStore; from src.utils import load_config; \\")
print("       vs = VectorStore(config=load_config()); \\")
print("       print(f'Count: {vs.count()}'); \\")
print("       r = vs.query('test', n_results=1); \\")
print("       print('✅ Query OK' if r['documents'][0] else '❌ Query failed')\"")
print("\n2. Transform queries:")
print("   python main.py transform all \"What is my portfolio allocation?\"")
print("\n3. Generate trajectories:")
print("   python main.py generate data/output/transformations/<latest>/transformations.jsonl")
print("\n" + "="*80)