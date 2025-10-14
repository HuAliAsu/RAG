import requests
import json
from typing import List, Optional

from src.utils.logger import logger
from src.core.config_manager import config_manager

class OllamaEmbedder:
    """
    A service to connect to an Ollama instance and generate embeddings.
    """
    def __init__(self):
        self.base_url = config_manager.get("Ollama", "base_url", fallback="http://localhost:11434")
        self.model = config_manager.get("Ollama", "embedding_model", fallback="embeddinggemma")
        self.timeout = config_manager.get_int("Ollama", "timeout", fallback=30)
        self.max_retries = config_manager.get_int("Ollama", "max_retries", fallback=3)
        self.api_url = f"{self.base_url}/api/embeddings"
        self.available = self.check_connection()

    def check_connection(self) -> bool:
        """
        Checks if the Ollama service is available and the specified model exists.
        """
        logger.info(f"Checking connection to Ollama at {self.base_url}...")
        try:
            # Check if the service is running
            response = requests.get(self.base_url, timeout=5)
            response.raise_for_status()
            logger.info("Ollama service is running.")

            # Check if the embedding model is available
            models_response = requests.get(f"{self.base_url}/api/tags")
            models_response.raise_for_status()
            models = models_response.json().get('models', [])

            if any(self.model in m['name'] for m in models):
                logger.info(f"Embedding model '{self.model}' is available on Ollama.")
                return True
            else:
                logger.error(f"Embedding model '{self.model}' not found on Ollama instance.")
                logger.error(f"Available models: {[m['name'] for m in models]}")
                logger.error(f"Please pull the model using: 'ollama pull {self.model}'")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}. Error: {e}")
            logger.error("Please ensure Ollama is running and accessible.")
            return False

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding for a single piece of text.
        """
        if not self.available:
            logger.warning("Ollama not available. Cannot generate embedding.")
            return None

        return self.get_embeddings([text])[0] if self.get_embeddings([text]) else None

    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generates embeddings for a list of texts with retry logic.
        """
        if not self.available:
            logger.warning("Ollama not available. Cannot generate embeddings.")
            return None

        if not texts:
            return []

        logger.info(f"Requesting embeddings for {len(texts)} text(s) using model '{self.model}'.")

        embeddings = []
        for text in texts:
            attempt = 0
            while attempt < self.max_retries:
                try:
                    payload = {
                        "model": self.model,
                        "prompt": text
                    }
                    response = requests.post(
                        self.api_url,
                        data=json.dumps(payload),
                        timeout=self.timeout
                    )
                    response.raise_for_status()

                    # Ollama returns one embedding per request
                    result = response.json()
                    if "embedding" in result:
                        embeddings.append(result["embedding"])
                        break  # Success, move to next text
                    else:
                        logger.error(f"Unexpected response from Ollama: {result}")
                        attempt += 1

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to get embedding (attempt {attempt + 1}/{self.max_retries}): {e}")
                    attempt += 1

            if attempt == self.max_retries:
                logger.error(f"Failed to get embedding for a text after {self.max_retries} retries.")
                # We append None to maintain the list size, or we could skip.
                # Skipping seems better to avoid downstream errors.
                continue

        if len(embeddings) != len(texts):
            logger.error("Could not generate embeddings for all provided texts.")
            return None

        logger.info(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings

# Singleton instance for the application
ollama_embedder = OllamaEmbedder()