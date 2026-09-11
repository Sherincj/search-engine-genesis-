"""
Index builder — batch construction of a field-aware inverted index.

Phase 8 update: now builds a ``FieldIndex`` with title and body fields::

    builder = FieldIndexBuilder()
    builder.add_document(1, title_tokens=["python"], body_tokens=["web"])
    builder.add_document(2, title_tokens=["database"], body_tokens=["sql"])
    index = builder.build()

Or in one step::

    index = FieldIndexBuilder.from_documents([
        (1, ["python"], ["web", "dev"]),
        (2, ["database"], ["sql", "storage"]),
    ])

The old ``IndexBuilder`` is preserved for backward compatibility with
existing code that only needs a single-field ``InvertedIndex``.
"""

from __future__ import annotations

from app.indexing.field_index import FieldIndex
from app.indexing.inverted_index import InvertedIndex


class IndexBuilder:
    """
    Accumulates documents and produces a complete ``InvertedIndex``.

    Preserved for backward compatibility with Phase 4–7 code.
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


class FieldIndexBuilder:
    """
    Accumulates documents and produces a ``FieldIndex``.

    Example::

        builder = FieldIndexBuilder()
        builder.add_document(1, ["python"], ["web", "dev"])
        index = builder.build()
    """

    def __init__(self) -> None:
        self._index = FieldIndex()
        self._document_count = 0

    def add_document(
        self,
        document_id: int,
        title_tokens: list[str] | None = None,
        body_tokens: list[str] | None = None,
    ) -> None:
        """
        Add a document's title and body tokens.

        Parameters
        ----------
        document_id : int
            Unique identifier for the document.
        title_tokens : list[str] | None
            Pre-normalised title tokens.
        body_tokens : list[str] | None
            Pre-normalised body tokens.
        """
        self._index.add_document(
            document_id,
            title_tokens=title_tokens,
            body_tokens=body_tokens,
        )
        self._document_count += 1

    def build(self) -> FieldIndex:
        """Return the constructed field index."""
        return self._index

    @staticmethod
    def from_documents(
        documents: list[tuple[int, list[str], list[str]]],
    ) -> FieldIndex:
        """
        Build a field index from ``(doc_id, title_tokens, body_tokens)`` triples.

        Example::

            index = FieldIndexBuilder.from_documents([
                (1, ["python"], ["web", "dev"]),
                (2, ["database"], ["sql", "storage"]),
            ])
        """
        builder = FieldIndexBuilder()
        for doc_id, title_tokens, body_tokens in documents:
            builder.add_document(doc_id, title_tokens, body_tokens)
        return builder.build()
