"""
Document indexer — coordinates the document → tokenise → index pipeline.

The ``DocumentIndexer`` is the single entry-point that future crawlers,
document pipelines, and API endpoints should call to index a document::

    Document
       ↓
    tokenize(document.text)
       ↓
    tokens
       ↓
    index.add_document(id, tokens)
       ↓
    IndexingResult

It does NOT:

* perform ranking or search
* modify BM25 or TF-IDF
* crawl URLs
* generate embeddings
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.indexing.inverted_index import InvertedIndex
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
    Coordinates tokenisation and index insertion for documents.

    Parameters
    ----------
    index : InvertedIndex
        The inverted index to populate.
    tokenizer : Tokenizer | None
        Custom tokeniser instance.  Defaults to the standard config
        (lowercase, Unicode normalisation, **no** stop-word removal).

    Usage::

        indexer = DocumentIndexer(index)
        result = indexer.index_document(doc_id=1, text="Python web dev")
    """

    def __init__(
        self,
        index: InvertedIndex,
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
    def index(self) -> InvertedIndex:
        """The underlying inverted index."""
        return self._index

    def index_document(
        self,
        document_id: int,
        text: str,
    ) -> IndexingResult:
        """
        Tokenise *text* and add the tokens to the inverted index.

        If the document was previously indexed, the old postings are
        replaced automatically (safe for re-indexing / updates).

        Parameters
        ----------
        document_id : int
            Database primary key of the document.
        text : str
            Raw document text.

        Returns
        -------
        IndexingResult
            Summary of what was indexed.
        """
        start = time.perf_counter()

        tokens = self._tokenizer.tokenize(text)
        self._index.add_document(document_id, tokens)

        elapsed_ms = (time.perf_counter() - start) * 1000

        unique_terms = len(set(tokens))

        logger.debug(
            "Indexed document %d: %d tokens, %d unique terms (%.1f ms)",
            document_id,
            len(tokens),
            unique_terms,
            elapsed_ms,
        )

        return IndexingResult(
            document_id=document_id,
            token_count=len(tokens),
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
        documents: list[tuple[int, str]],
    ) -> list[IndexingResult]:
        """
        Index multiple documents.

        Parameters
        ----------
        documents : list[tuple[int, str]]
            List of ``(document_id, text)`` pairs.

        Returns
        -------
        list[IndexingResult]
            One result per document.
        """
        results = []
        for doc_id, text in documents:
            results.append(self.index_document(doc_id, text))
        logger.info(
            "Bulk indexed %d documents (%d total in index)",
            len(results),
            self._index.get_total_documents(),
        )
        return results
