"""Application service for the independent persistent semantic index."""

from __future__ import annotations

from app.config.settings import settings
from app.embeddings.sentence_transformer import SentenceTransformerProvider
from app.indexing.semantic_indexer import SemanticIndexer
from app.search.semantic_search_engine import SemanticSearchEngine
from app.vector_index.vector_index import VectorIndex
from app.vector_index.vector_storage import VectorStorage, VectorStorageError


class SemanticIndexServiceError(Exception):
    """Raised when the configured semantic index cannot be used safely."""


class SemanticIndexService:
    """Own the provider, vector index, persistence, and semantic engine."""

    def __init__(self) -> None:
        self.provider = SentenceTransformerProvider(
            model_name=settings.EMBEDDING_MODEL,
            normalize=settings.EMBEDDING_NORMALIZE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
        self.storage = VectorStorage(
            settings.VECTOR_INDEX_PATH,
            version=settings.VECTOR_INDEX_VERSION,
        )
        try:
            dimension = self.provider.dimension
            if self.storage.exists():
                self.index, _ = self.storage.load(
                    expected_model=self.provider.model_name,
                    expected_dimension=dimension,
                    expected_normalize=settings.EMBEDDING_NORMALIZE,
                    expected_version=settings.VECTOR_INDEX_VERSION,
                )
            else:
                self.index = VectorIndex(dimension=dimension)
        except (ImportError, OSError, ValueError, VectorStorageError) as exc:
            raise SemanticIndexServiceError(str(exc)) from exc

        self.indexer = SemanticIndexer(self.index, self.provider)
        self.engine = SemanticSearchEngine(self.index, self.provider)

    def _save(self) -> None:
        self.index.validate()
        self.storage.save(
            self.index,
            model_name=self.provider.model_name,
            normalize=settings.EMBEDDING_NORMALIZE,
        )

    def index_document(self, document_id: int, title: str | None, text: str) -> None:
        self.indexer.index_document(document_id, title, text)
        self._save()

    def update_document(self, document_id: int, title: str | None, text: str) -> None:
        self.indexer.update_document(document_id, title, text)
        self._save()

    def remove_document(self, document_id: int) -> bool:
        removed = self.indexer.remove_document(document_id)
        if removed:
            self._save()
        return removed


_semantic_index_service: SemanticIndexService | None = None


def get_semantic_index_service() -> SemanticIndexService:
    """Return the configured semantic service, creating it lazily."""
    global _semantic_index_service
    if not settings.EMBEDDING_ENABLED:
        raise SemanticIndexServiceError("Semantic search is disabled")
    if _semantic_index_service is None:
        _semantic_index_service = SemanticIndexService()
    return _semantic_index_service


def set_semantic_index_service(service: SemanticIndexService | None) -> None:
    """Override/reset the singleton in tests or after configuration changes."""
    global _semantic_index_service
    _semantic_index_service = service
