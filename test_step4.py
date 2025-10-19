"""
Test script for ChromaDB Manager (Step 4).
Tests vector storage, retrieval, and integration with Bedrock.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.utils import load_config, setup_logger, get_logger
from src.core import BedrockProvider, ChromaDBManager

def test_initialization():
    """Test ChromaDB manager initialization."""
    print("\n" + "="*60)
    print("TEST 1: ChromaDB Initialization")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        config = load_config()
        
        manager = ChromaDBManager(
            persist_directory=config.chromadb.persist_directory,
            collection_name=config.chromadb.collection_name,
            distance_metric=config.chromadb.distance_metric
        )
        
        logger.info(f"✅ Manager initialized: {manager}")
        
        # Get stats
        stats = manager.get_stats()
        logger.info(f"Collection stats: {stats['total_documents']} documents")
        
        print("✅ Initialization test passed\n")
        return manager
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def test_add_documents(manager: ChromaDBManager, provider: BedrockProvider):
    """Test adding documents with embeddings."""
    print("="*60)
    print("TEST 2: Add Documents with Embeddings")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # Sample documents
        documents = [
            "Artificial intelligence is transforming how we work and live.",
            "Machine learning algorithms can learn from data without explicit programming.",
            "Deep learning uses neural networks inspired by the human brain.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret visual information."
        ]
        
        # Generate embeddings using Bedrock
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = provider.generate_embeddings_batch(documents)
        
        # Create metadata
        metadatas = [
            {"topic": "AI", "complexity": "simple", "type": "overview"},
            {"topic": "ML", "complexity": "medium", "type": "technical"},
            {"topic": "DL", "complexity": "complex", "type": "technical"},
            {"topic": "NLP", "complexity": "medium", "type": "technical"},
            {"topic": "CV", "complexity": "medium", "type": "technical"}
        ]
        
        # Add to ChromaDB
        ids = manager.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(ids)} documents")
        print(f"\n✅ Added {len(ids)} documents with embeddings")
        print(f"Document IDs: {ids[:2]}... (showing first 2)\n")
        
        print("✅ Add documents test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Add documents failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_similarity_search(manager: ChromaDBManager, provider: BedrockProvider):
    """Test similarity search."""
    print("="*60)
    print("TEST 3: Similarity Search")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # Query text
        query_text = "How do neural networks work in deep learning?"
        
        logger.info(f"Query: {query_text}")
        
        # Generate query embedding
        query_embedding = provider.generate_embedding(query_text)
        
        # Search
        results = manager.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Display results
        print(f"\nQuery: {query_text}")
        print(f"\nTop 3 similar documents:")
        
        for i, (doc, distance) in enumerate(zip(
            results['documents'][0],
            results['distances'][0]
        ), 1):
            print(f"\n{i}. Distance: {distance:.4f}")
            print(f"   Document: {doc[:80]}...")
        
        print("\n✅ Similarity search test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Similarity search failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_filtering(manager: ChromaDBManager, provider: BedrockProvider):
    """Test metadata filtering."""
    print("="*60)
    print("TEST 4: Metadata Filtering")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        query_text = "Tell me about AI technology"
        query_embedding = provider.generate_embedding(query_text)
        
        # Filter by complexity
        results = manager.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"complexity": "medium"}
        )
        
        print(f"\nQuery: {query_text}")
        print(f"Filter: complexity = 'medium'")
        print(f"\nFound {len(results['documents'][0])} matching documents:")
        
        for i, (doc, meta) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        ), 1):
            print(f"\n{i}. Topic: {meta.get('topic')}, Complexity: {meta.get('complexity')}")
            print(f"   {doc[:60]}...")
        
        print("\n✅ Metadata filtering test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Metadata filtering failed: {e}\n")
        return False


def test_batch_operations(manager: ChromaDBManager, provider: BedrockProvider):
    """Test batch document operations."""
    print("="*60)
    print("TEST 5: Batch Operations")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # Create batch of documents
        batch_docs = [
            f"This is test document number {i} about various AI topics."
            for i in range(10)
        ]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(batch_docs)} documents...")
        batch_embeddings = provider.generate_embeddings_batch(batch_docs)
        
        # Create metadata
        batch_metadata = [
            {"batch": "test", "index": i, "type": "generated"}
            for i in range(10)
        ]
        
        # Add in batch
        ids = manager.add_documents_batch(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadata,
            batch_size=5
        )
        
        print(f"\n✅ Added {len(ids)} documents in batches")
        print(f"Total documents in collection: {manager.count()}\n")
        
        print("✅ Batch operations test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Batch operations failed: {e}\n")
        return False


def test_collection_stats(manager: ChromaDBManager):
    """Test collection statistics."""
    print("="*60)
    print("TEST 6: Collection Statistics")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        stats = manager.get_stats()
        
        print(f"\nCollection Statistics:")
        print(f"  Name: {stats['collection_name']}")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Distance Metric: {stats['distance_metric']}")
        print(f"  Metadata Keys: {', '.join(stats['metadata_keys'])}")
        print(f"  Persist Directory: {stats['persist_directory']}")
        
        print("\n✅ Statistics test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "🗄️"*30)
    print("CHROMADB MANAGER TEST SUITE")
    print("🗄️"*30)
    
    # Setup logger
    setup_logger("INFO", "logs/test_chromadb.log")
    
    # Initialize config
    config = load_config()
    
    # Initialize Bedrock provider (for embeddings)
    print("\nInitializing Bedrock Provider...")
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        embedding_model_id=config.bedrock.embedding_model_id,
        region=config.bedrock.region
    )
    print("✅ Bedrock provider ready\n")
    
    # Test 1: Initialization
    manager = test_initialization()
    
    if not manager:
        print("❌ Cannot proceed without manager initialization")
        return
    
    # Clear any existing data for clean test
    print("Cleaning collection for fresh test...")
    manager.clear_collection()
    print("✅ Collection cleared\n")
    
    # Run tests
    results = []
    
    results.append(("Add Documents", test_add_documents(manager, provider)))
    results.append(("Similarity Search", test_similarity_search(manager, provider)))
    results.append(("Metadata Filtering", test_metadata_filtering(manager, provider)))
    results.append(("Batch Operations", test_batch_operations(manager, provider)))
    results.append(("Collection Stats", test_collection_stats(manager)))
    
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
        print("🎉 ALL CHROMADB TESTS PASSED!")
        print("="*60)
        print(f"\nFinal collection size: {manager.count()} documents")
        print("\nStep 4: ChromaDB Manager ✅ COMPLETE\n")
        print("Next: Implement Multi-modal PDF Parser (Step 5)")
    else:
        print("⚠️  SOME CHROMADB TESTS FAILED")
        print("="*60)
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()