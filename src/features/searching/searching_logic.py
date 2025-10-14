from typing import List, Dict, Any, Optional

from src.utils.logger import logger
from src.core.embedder import ollama_embedder
from src.core.vector_db import ChromaDBManager

class SearchLogic:
    """
    Handles the logic for performing semantic search against the vector database.
    """
    def __init__(self):
        self.db_manager = ChromaDBManager()
        self.embedder = ollama_embedder

    def list_available_collections(self) -> List[str]:
        """
        Returns a list of available collection names, stripping the prefix.
        """
        if not self.db_manager.client:
            logger.warning("ChromaDB client not available. Cannot list collections.")
            return []

        prefix = self.db_manager.collection_prefix
        prefix_len = len(prefix)

        try:
            collections = self.db_manager.list_collections()
            # Return only the names without the internal prefix
            return [name[prefix_len:] for name in collections if name.startswith(prefix)]
        except Exception as e:
            logger.error(f"Failed to list ChromaDB collections: {e}")
            return []

    def perform_search(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Executes the end-to-end search process.
        1. Generates embedding for the query.
        2. Queries the vector database.
        3. Formats the results for the UI.
        """
        logger.info(f"Performing search in '{collection_name}' for query: '{query_text[:50]}...'")

        # 1. Generate query embedding
        if not self.embedder.available:
            logger.error("Ollama embedder is not available. Cannot perform search.")
            return None

        query_embedding = self.embedder.get_embedding(query_text)
        if not query_embedding:
            logger.error("Failed to generate embedding for the query.")
            return None
        logger.debug("Query embedding generated successfully.")

        # 2. Query the vector database
        raw_results = self.db_manager.search(
            collection_name=collection_name, # The logic adds the prefix internally
            query_embedding=query_embedding,
            n_results=n_results
        )

        if not raw_results or not raw_results.get('ids', [[]])[0]:
            logger.warning("Search returned no results from the database.")
            return []

        # 3. Format the results
        formatted_results = self._format_results(raw_results)
        logger.info(f"Search successful. Found {len(formatted_results)} formatted results.")

        return formatted_results

    def _format_results(self, raw_results: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Transforms the raw dictionary from ChromaDB into a list of
        dictionaries suitable for the UI.
        """
        formatted = []

        # ChromaDB returns lists of lists, even for a single query
        ids = raw_results.get('ids', [[]])[0]
        distances = raw_results.get('distances', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        documents = raw_results.get('documents', [[]])[0]

        for i, doc_id in enumerate(ids):
            distance = distances[i]
            # Convert distance to similarity score (for cosine)
            similarity = 1 - distance if distance is not None else 0

            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

            result = {
                "id": doc_id,
                "score": similarity,
                "text": documents[i] if documents and i < len(documents) else "متن موجود نیست",
                "source": metadata.get("source_document", "نامشخص"),
                "chunk_id": metadata.get("chunk_id", "نامشخص"),
                "metadata": metadata # Keep all metadata for detailed view
            }
            formatted.append(result)

        return formatted

# Singleton instance for the application to use
search_logic = SearchLogic()