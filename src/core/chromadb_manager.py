"""
ChromaDB Manager for vector storage and similarity search.
Handles document chunks, embeddings, and metadata.

FIXED VERSION: Uses PersistentClient for proper data persistence
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..utils import get_logger

logger = get_logger(__name__)


class ChromaDBManager:
    """Manage ChromaDB collections for document storage and retrieval."""
    
    def __init__(
        self,
        persist_directory: str = "data/chromadb",
        collection_name: str = "document_chunks",
        distance_metric: str = "cosine",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            distance_metric: Distance metric (cosine, l2, ip)
            embedding_function: Custom embedding function (optional)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = self._initialize_client()
        
        # Initialize collection
        self.collection = self._get_or_create_collection(embedding_function)
        
        logger.info(
            f"Initialized ChromaDBManager: collection='{collection_name}', "
            f"metric={distance_metric}"
        )
    
    def _initialize_client(self) -> chromadb.ClientAPI:
        """Initialize ChromaDB client with persistence.
        
        FIXED: Uses PersistentClient instead of Client for proper persistence.
        """
        try:
            # Use PersistentClient for data persistence
            client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            logger.debug(f"ChromaDB PersistentClient initialized at: {self.persist_directory}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _get_or_create_collection(
        self,
        embedding_function: Optional[Any] = None
    ) -> chromadb.Collection:
        """Get existing collection or create new one."""
        try:
            # Map distance metric to ChromaDB space
            space_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip"
            }
            space = space_map.get(self.distance_metric, "cosine")
            
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": space},
                embedding_function=embedding_function
            )
            
            logger.debug(
                f"Collection '{self.collection_name}' ready "
                f"({collection.count()} documents)"
            )
            return collection
            
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents with embeddings to collection.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs (auto-generated if None)
            
        Returns:
            List of document IDs
        """
        n_docs = len(documents)
        
        if len(embeddings) != n_docs:
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of documents ({n_docs})"
            )
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in range(n_docs)]
        
        # Use empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in range(n_docs)]
        
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {n_docs} documents to collection")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def add_documents_batch(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents in batches for better performance.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            batch_size: Number of documents per batch
            
        Returns:
            List of all document IDs
        """
        all_ids = []
        n_docs = len(documents)
        
        logger.info(f"Adding {n_docs} documents in batches of {batch_size}")
        
        for i in range(0, n_docs, batch_size):
            batch_end = min(i + batch_size, n_docs)
            
            batch_docs = documents[i:batch_end]
            batch_embs = embeddings[i:batch_end]
            batch_meta = metadatas[i:batch_end] if metadatas else None
            
            batch_ids = self.add_documents(
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_meta
            )
            
            all_ids.extend(batch_ids)
            logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch_ids)} docs")
        
        logger.info(f"Completed batch addition: {len(all_ids)} total documents")
        return all_ids
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, str]] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Any]:
        """
        Query collection for similar documents.
        
        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results per query
            where: Metadata filter conditions
            where_document: Document content filter conditions
            include: What to include in results
            
        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
            n_queries = len(query_embeddings)
            logger.debug(
                f"Queried {n_queries} embeddings, "
                f"retrieved up to {n_results} results each"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def query_by_text(
        self,
        query_texts: List[str],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query collection using text (requires embedding function in collection).
        
        Args:
            query_texts: List of query texts
            n_results: Number of results per query
            where: Metadata filter conditions
            
        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.debug(f"Text query: {len(query_texts)} queries")
            return results
            
        except Exception as e:
            logger.error(f"Text query failed: {e}")
            raise
    
    def get_by_ids(
        self,
        ids: List[str],
        include: List[str] = ["documents", "metadatas", "embeddings"]
    ) -> Dict[str, Any]:
        """
        Get documents by IDs.
        
        Args:
            ids: List of document IDs
            include: What to include in results
            
        Returns:
            Documents matching the IDs
        """
        try:
            results = self.collection.get(
                ids=ids,
                include=include
            )
            
            logger.debug(f"Retrieved {len(ids)} documents by ID")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get documents by ID: {e}")
            raise
    
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Update existing documents.
        
        Args:
            ids: List of document IDs to update
            documents: Optional updated document texts
            embeddings: Optional updated embeddings
            metadatas: Optional updated metadata
        """
        try:
            self.collection.update(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Updated {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            raise
    
    def delete_documents(self, ids: List[str]):
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def delete_by_metadata(self, where: Dict[str, Any]):
        """
        Delete documents by metadata filter.
        
        Args:
            where: Metadata filter conditions
        """
        try:
            self.collection.delete(where=where)
            logger.info(f"Deleted documents matching filter: {where}")
            
        except Exception as e:
            logger.error(f"Failed to delete by metadata: {e}")
            raise
    
    def count(self) -> int:
        """Get total number of documents in collection."""
        count = self.collection.count()
        logger.debug(f"Collection contains {count} documents")
        return count
    
    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """
        Peek at first few documents in collection.
        
        Args:
            limit: Number of documents to peek
            
        Returns:
            First documents in collection
        """
        try:
            results = self.collection.peek(limit=limit)
            logger.debug(f"Peeked at {limit} documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to peek: {e}")
            raise
    
    def clear_collection(self):
        """Delete all documents from collection."""
        try:
            # Get all IDs
            all_docs = self.collection.get()
            ids = all_docs.get('ids', [])
            
            if ids:
                self.collection.delete(ids=ids)
                logger.warning(f"Cleared collection: deleted {len(ids)} documents")
            else:
                logger.info("Collection already empty")
                
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        count = self.count()
        
        # Sample a few documents to check metadata
        sample = self.peek(limit=5)
        metadatas = sample.get('metadatas', [])
        
        # Get unique metadata keys
        metadata_keys = set()
        for meta in metadatas:
            if meta:
                metadata_keys.update(meta.keys())
        
        stats = {
            'collection_name': self.collection_name,
            'total_documents': count,
            'distance_metric': self.distance_metric,
            'metadata_keys': list(metadata_keys),
            'persist_directory': str(self.persist_directory)
        }
        
        logger.info(f"Collection stats: {count} documents")
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChromaDBManager(collection='{self.collection_name}', "
            f"docs={self.count()}, metric={self.distance_metric})"
        )


if __name__ == "__main__":
    # Test the manager
    from ..utils import setup_logger
    
    setup_logger("INFO")
    
    # Initialize manager
    manager = ChromaDBManager(
        persist_directory="data/chromadb_test",
        collection_name="test_collection"
    )
    
    print(f"\n✅ Manager initialized: {manager}\n")
    
    # Test adding documents
    test_docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text."
    ]
    
    # Mock embeddings (in real use, these would come from BedrockProvider)
    test_embeddings = [
        [0.1] * 1024,
        [0.2] * 1024,
        [0.3] * 1024
    ]
    
    test_metadata = [
        {"topic": "ML", "complexity": "simple"},
        {"topic": "DL", "complexity": "medium"},
        {"topic": "NLP", "complexity": "simple"}
    ]
    
    ids = manager.add_documents(
        documents=test_docs,
        embeddings=test_embeddings,
        metadatas=test_metadata
    )
    
    print(f"✅ Added {len(ids)} documents\n")
    
    # Get stats
    stats = manager.get_stats()
    print(f"✅ Stats: {stats}\n")
    
    # Query
    query_emb = [[0.15] * 1024]
    results = manager.query(query_embeddings=query_emb, n_results=2)
    print(f"✅ Query returned {len(results['documents'][0])} results\n")
    
    print("="*60)
    print("✅ All tests completed!")
    print("="*60)