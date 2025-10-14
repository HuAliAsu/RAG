import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional

from src.utils.logger import logger
from src.core.config_manager import config_manager
from src.core.embedder import ollama_embedder

class ChromaDBManager:
    """
    Manages interactions with a ChromaDB vector store.
    """
    def __init__(self):
        self.persist_directory = config_manager.get("ChromaDB", "persist_directory", fallback="./data/vector_stores")
        self.collection_prefix = config_manager.get("ChromaDB", "collection_prefix", fallback="rag_")
        self.distance_metric = config_manager.get("ChromaDB", "distance_metric", fallback="cosine")

        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"ChromaDB client initialized. Persisting data to '{self.persist_directory}'.")
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB client at '{self.persist_directory}': {e}")
            self.client = None

        # This embedding function is a fallback if pre-computed embeddings aren't provided.
        # However, our pipeline will generate embeddings beforehand.
        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            url=ollama_embedder.api_url.replace("/api/embeddings", ""), # OllamaEF needs the base URL
            model_name=ollama_embedder.model
        ) if ollama_embedder.available else None

    def get_or_create_collection(self, collection_name: str) -> Optional[chromadb.Collection]:
        """
        Retrieves a collection or creates it if it doesn't exist.
        """
        if not self.client:
            logger.error("ChromaDB client is not available.")
            return None

        full_collection_name = f"{self.collection_prefix}{collection_name}"
        try:
            collection = self.client.get_or_create_collection(
                name=full_collection_name,
                metadata={"hnsw:space": self.distance_metric} # Cosine distance
            )
            logger.info(f"Successfully accessed or created collection: '{full_collection_name}'.")
            return collection
        except Exception as e:
            logger.error(f"Failed to get or create collection '{full_collection_name}': {e}")
            return None

    def add_to_collection(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """
        Adds documents and their embeddings to a specified collection.
        """
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            return False

        try:
            # Add data in batches to avoid overwhelming the system
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                logger.info(f"Added batch of {len(batch_ids)} documents to '{collection.name}'.")

            return True
        except Exception as e:
            logger.error(f"Failed to add documents to collection '{collection.name}': {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Optional[Dict[str, List[Any]]]:
        """
        Performs a similarity search in a collection.
        """
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            return None

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            logger.info(f"Search completed in '{collection.name}' with {len(results.get('ids', [[]])[0])} results.")
            return results
        except Exception as e:
            logger.error(f"Search failed in collection '{collection.name}': {e}")
            return None

    def list_collections(self) -> List[str]:
        """Lists all collections in the database."""
        if not self.client:
            return []
        return [col.name for col in self.client.list_collections()]

    def delete_collection(self, collection_name: str) -> bool:
        """Deletes a collection."""
        full_collection_name = f"{self.collection_prefix}{collection_name}"
        try:
            self.client.delete_collection(name=full_collection_name)
            logger.info(f"Collection '{full_collection_name}' deleted successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{full_collection_name}': {e}")
            return False