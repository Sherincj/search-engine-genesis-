"""
Document indexer — coordinates the document → tokenise → index pipeline.

Phase 8 update: now indexes documents into a ``FieldIndex`` with
separate title and body fields::

    Document
       ├── title → tokenize → title index
       └── text  → tokenize → body index

The ``DocumentIndexer`` remains the single entry-point that future
crawlers, document pipelines, and API endpoints should call.

It does NOT:

* perform ranking or search
* modify BM25 or TF-IDF
* crawl URLs
* generate embeddings
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.indexing.field_index import FieldIndex
from app.text_processing import Tokenizer, TokenizerConfig
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndexingResult:
    """Metadata returned after indexing a document."""

    document_id: int
    token_count: int
    unique_terms: int
    duration_ms: float


class DocumentIndexer:
    """
    Coordinates tokenisation and field-aware index insertion.

    Parameters
    ----------
    index : FieldIndex
        The field-aware inverted index to populate.
    tokenizer : Tokenizer | None
        Custom tokeniser instance.  Defaults to the standard config
        (lowercase, Unicode normalisation, **no** stop-word removal).

    Usage::

        indexer = DocumentIndexer(field_index)
        result = indexer.index_document(
            doc_id=1, title="Python Guide", text="Python web dev"
        )
    """

    def __init__(
        self,
        index: FieldIndex,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._index = index
        self._tokenizer = tokenizer or Tokenizer(
            TokenizerConfig(
                lowercase=True,
                normalize_unicode=True,
                remove_stopwords=False,
            )
        )

    @property
    def index(self) -> FieldIndex:
        """The underlying field-aware inverted index."""
        return self._index

    def index_document(
        self,
        document_id: int,
        text: str,
        *,
        title: str = "",
    ) -> IndexingResult:
        """
        Tokenise title and text, then add to the field-aware index.

        If the document was previously indexed, the old postings are
        replaced automatically (safe for re-indexing / updates).

        Parameters
        ----------
        document_id : int
            Database primary key of the document.
        text : str
            Raw document body text.
        title : str
            Raw document title (may be empty).

        Returns
        -------
        IndexingResult
            Summary of what was indexed.
        """
        start = time.perf_counter()

        title_tokens = self._tokenizer.tokenize(title) if title else []
        body_tokens = self._tokenizer.tokenize(text)

        self._index.add_document(
            document_id,
            title_tokens=title_tokens,
            body_tokens=body_tokens,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        all_tokens = title_tokens + body_tokens
        unique_terms = len(set(all_tokens))

        logger.debug(
            "Indexed document %d: %d title + %d body tokens, "
            "%d unique terms (%.1f ms)",
            document_id,
            len(title_tokens),
            len(body_tokens),
            unique_terms,
            elapsed_ms,
        )

        return IndexingResult(
            document_id=document_id,
            token_count=len(all_tokens),
            unique_terms=unique_terms,
            duration_ms=round(elapsed_ms, 2),
        )

    def remove_document(self, document_id: int) -> bool:
        """
        Remove a document from the index.

        Returns ``True`` if the document was present, ``False`` otherwise.
        """
        removed = self._index.remove_document(document_id)
        if removed:
            logger.debug("Removed document %d from index", document_id)
        return removed

    def bulk_index(
        self,
        documents: list[tuple[int, str, str]],
    ) -> list[IndexingResult]:
        """
        Index multiple documents.

        Parameters
        ----------
        documents : list[tuple[int, str, str]]
            List of ``(document_id, title, text)`` triples.

        Returns
        -------
        list[IndexingResult]
            One result per document.
        """
        results = []
        for doc_id, title, text in documents:
            results.append(
                self.index_document(doc_id, text, title=title)
            )
        logger.info(
            "Bulk indexed %d documents (%d total in index)",
            len(results),
            self._index.get_total_documents(),
        )
        return results
