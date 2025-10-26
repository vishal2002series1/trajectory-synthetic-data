from src.core import ChromaDBManager
from src.utils import load_config

config = load_config()
manager = ChromaDBManager(
    persist_directory=config.chromadb.persist_directory,
    collection_name="document_chunks"
)

print(f"Total documents: {manager.count()}")
print(f"Collection: {manager.collection.name}")