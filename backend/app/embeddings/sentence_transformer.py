"""
Sentence-transformer embedding provider — local model integration.

Uses the sentence-transformers library to generate embeddings locally.
The model is loaded lazily on first use.
"""

from __future__ import annotations

from app.embeddings.types import Embedding
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerProvider:
    """
    Local sentence-transformer embedding provider.

    Parameters
    ----------
    model_name : str
        Name of the sentence-transformers model to load.
    normalize : bool
        Whether to L2-normalize embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._normalize = normalize
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size
        self._model = None  # Lazy loading
        self._dimension: int | None = None

    def _load_model(self) -> None:
        """Load the model lazily."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._dimension = self._model.get_embedding_dimension()
        logger.info(
            "Model loaded: %s (dimension=%d)",
            self._model_name,
            self._dimension,
        )

    def embed_text(self, text: str) -> Embedding:
        """Embed a single text string."""
        self._load_model()
        embedding = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of texts."""
        if not texts:
            return []

        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
