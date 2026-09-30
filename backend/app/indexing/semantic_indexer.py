"""
Semantic indexer — manages document embeddings for semantic search.

Handles generating embeddings from documents and updating the VectorIndex.
"""

from __future__ import annotations

from app.embeddings.provider import EmbeddingProvider, build_embedding_text
from app.vector_index.vector_index import VectorIndex


class SemanticIndexer:
    """
    Manages vector index updates by embedding documents.

    Parameters
    ----------
    index : VectorIndex
        The target vector index to update.
    provider : EmbeddingProvider
        The provider used to embed documents.
    """

    def __init__(self, index: VectorIndex, provider: EmbeddingProvider) -> None:
        self._index = index
        self._provider = provider

    @property
    def index(self) -> VectorIndex:
        return self._index

    def index_document(self, doc_id: int, title: str | None, text: str) -> None:
        """
        Embed and index a document.

        Parameters
        ----------
        doc_id : int
            Document ID.
        title : str | None
            Document title.
        text : str
            Document body text.
        """
        embedding_text = build_embedding_text(title or "", text)
        if not embedding_text:
            return  # Nothing to embed

        vector = self._provider.embed_text(embedding_text)
        self._index.add(doc_id, vector)

    def update_document(self, doc_id: int, title: str | None, text: str) -> None:
        """
        Update an existing document's embedding. Same as index_document.
        """
        self.index_document(doc_id, title, text)

    def remove_document(self, doc_id: int) -> bool:
        """
        Remove a document from the vector index.

        Returns
        -------
        bool
            True if removed, False if not found.
        """
        return self._index.remove(doc_id)

    def bulk_index(self, documents: list[tuple[int, str | None, str]]) -> None:
        """
        Embed and index a batch of documents efficiently.

        Parameters
        ----------
        documents : list[tuple[int, str | None, str]]
            List of (id, title, text) tuples.
        """
        if not documents:
            return

        texts = []
        doc_ids = []
        for doc_id, title, text in documents:
            embedding_text = build_embedding_text(title or "", text)
            if embedding_text:
                texts.append(embedding_text)
                doc_ids.append(doc_id)

        if not texts:
            return

        vectors = self._provider.embed_documents(texts)
        if len(vectors) != len(doc_ids):
            raise ValueError("Embedding provider returned an unexpected number of vectors")
        for doc_id, vector in zip(doc_ids, vectors):
            self._index.add(doc_id, vector)
