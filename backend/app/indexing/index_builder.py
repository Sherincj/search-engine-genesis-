"""
Index builder — batch construction of an inverted index.

Provides a convenient builder interface for creating an index
from a collection of documents::

    builder = IndexBuilder()
    builder.add_document(1, ["python", "web"])
    builder.add_document(2, ["python", "api"])
    index = builder.build()

Or in one step::

    index = IndexBuilder.from_documents([
        (1, ["python", "web"]),
        (2, ["python", "api"]),
    ])
"""

from __future__ import annotations

from app.indexing.inverted_index import InvertedIndex


class IndexBuilder:
    """
    Accumulates documents and produces a complete ``InvertedIndex``.

    The builder is a thin coordination layer — the actual indexing
    logic lives in ``InvertedIndex.add_document()``.
    """

    def __init__(self) -> None:
        self._index = InvertedIndex()
        self._document_count = 0

    def add_document(self, document_id: int, tokens: list[str]) -> None:
        """
        Add a document's tokens to the index being built.

        Parameters
        ----------
        document_id : int
            Unique identifier for the document.
        tokens : list[str]
            Pre-normalised tokens from the tokeniser.
        """
        self._index.add_document(document_id, tokens)
        self._document_count += 1

    def build(self) -> InvertedIndex:
        """Return the constructed index."""
        return self._index

    @staticmethod
    def from_documents(
        documents: list[tuple[int, list[str]]],
    ) -> InvertedIndex:
        """
        Build an index from a list of ``(document_id, tokens)`` pairs.

        Example::

            index = IndexBuilder.from_documents([
                (1, ["python", "web"]),
                (2, ["javascript", "web"]),
            ])
        """
        builder = IndexBuilder()
        for doc_id, tokens in documents:
            builder.add_document(doc_id, tokens)
        return builder.build()
