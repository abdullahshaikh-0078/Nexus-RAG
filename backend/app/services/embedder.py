import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates vector embeddings using SentenceTransformers models."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generates embedding vector for a single text string."""
        model = self._get_model()
        vector = model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of text strings."""
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True, batch_size=32)
        return vectors.tolist()


# Global singleton instance
embedding_service = EmbeddingService()
