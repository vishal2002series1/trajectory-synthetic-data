"""
System Verification - Check Before Starting Workflow

Verifies all components are working before re-ingesting PDFs
"""

from pathlib import Path
from src.utils import load_config, setup_logger
import sys

print("\n" + "="*80)
print("SYSTEM VERIFICATION CHECK")
print("="*80)

setup_logger("INFO")
all_checks_passed = True

# Check 1: Config file
print("\n[1/6] Checking config.yaml...")
try:
    config = load_config()
    print(f"   ✅ Config loaded")
    print(f"      Collection name: {config.chromadb.collection_name}")
    print(f"      Model ID: {config.bedrock.model_id}")
    print(f"      Embedding model: {config.bedrock.embedding_model_id}")
except Exception as e:
    print(f"   ❌ Config error: {e}")
    all_checks_passed = False

# Check 2: PDF directory
print("\n[2/6] Checking PDF directory...")
pdf_dir = Path("data/pdfs")
if pdf_dir.exists():
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if pdf_files:
        print(f"   ✅ Found {len(pdf_files)} PDF files")
        for pdf in pdf_files[:3]:
            print(f"      - {pdf.name}")
        if len(pdf_files) > 3:
            print(f"      ... and {len(pdf_files) - 3} more")
    else:
        print(f"   ⚠️  Directory exists but no PDFs found")
        print(f"      Add PDFs to: {pdf_dir.absolute()}")
        all_checks_passed = False
else:
    print(f"   ❌ Directory not found: {pdf_dir}")
    print(f"      Create it with: mkdir -p {pdf_dir}")
    all_checks_passed = False

# Check 3: Core components
print("\n[3/6] Checking core components...")
try:
    from src.core import BedrockProvider, VectorStore, PDFParser
    print(f"   ✅ All core components importable")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    all_checks_passed = False

# Check 4: Bedrock connectivity (optional - may require AWS creds)
print("\n[4/6] Checking AWS Bedrock connection...")
try:
    from src.core import BedrockProvider
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        embedding_model_id=config.bedrock.embedding_model_id,
        region=config.bedrock.region
    )
    # Try a simple embedding
    test_emb = provider.generate_embedding("test")
    if len(test_emb) > 0:
        print(f"   ✅ Bedrock connected (embedding dimension: {len(test_emb)})")
    else:
        print(f"   ⚠️  Bedrock connected but empty embedding")
except Exception as e:
    print(f"   ⚠️  Bedrock connection issue: {e}")
    print(f"      This may be OK if AWS credentials aren't configured yet")
    # Don't fail the check for this

# Check 5: VectorStore initialization
print("\n[5/6] Checking VectorStore...")
try:
    from src.core import VectorStore
    vs = VectorStore(config=config)
    count = vs.count()
    print(f"   ✅ VectorStore initialized")
    print(f"      Collection: {config.chromadb.collection_name}")
    print(f"      Documents: {count}")
    
    if count == 0:
        print(f"      ℹ️  Collection is empty (expected for fresh collection)")
except Exception as e:
    print(f"   ❌ VectorStore error: {e}")
    all_checks_passed = False

# Check 6: CLI commands
print("\n[6/6] Checking CLI structure...")
try:
    cli_dir = Path("src/cli")
    if cli_dir.exists():
        cli_files = list(cli_dir.glob("*_commands.py"))
        print(f"   ✅ CLI directory exists")
        print(f"      Found {len(cli_files)} command modules")
        for f in cli_files:
            print(f"      - {f.stem}")
    else:
        print(f"   ⚠️  CLI directory not found: {cli_dir}")
except Exception as e:
    print(f"   ⚠️  CLI check error: {e}")

# Check main.py
main_file = Path("main.py")
if main_file.exists():
    print(f"   ✅ main.py exists")
else:
    print(f"   ❌ main.py not found")
    all_checks_passed = False

# Final summary
print("\n" + "="*80)
if all_checks_passed:
    print("✅ ALL CRITICAL CHECKS PASSED")
    print("="*80)
    print("\n🎯 READY TO PROCEED")
    print("\nNext steps:")
    print("   1. Update config: nano config/config.yaml")
    print("      Change: collection_name: 'document_chunks_v2'")
    print("   2. Ingest PDFs: python ingest_pdfs_fresh.py")
    print("   3. Follow: COMPLETE_WORKFLOW_PLAN.md")
else:
    print("⚠️  SOME CHECKS FAILED")
    print("="*80)
    print("\nFix the issues above before proceeding")

print("="*80)