"""
Search service — coordinates the search engine with database retrieval.

This is the main orchestration layer between:

* ``SearchEngine`` (pure search: parse → index → rank)
* ``DocumentRepository`` (SQLite document retrieval)
* ``IndexStorage`` (index persistence)
* ``DocumentIndexer`` (index rebuilding)

Architecture::

    SearchService
        ├── SearchEngine  (field-aware inverted index lookups)
        ├── DocumentRepository  (SQLite)
        └── IndexStorage  (load / save)
"""

from __future__ import annotations

import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.database.repositories import DocumentRepository
from app.indexing import (
    DocumentIndexer,
    FieldIndex,
    IndexStorage,
    IndexStorageError,
)
from app.models.search import SearchResponse, SearchResultItem
from app.search.search_engine import SearchEngine
from app.search.snippet_generator import SnippetGenerator
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SearchService:
    """
    Coordinates search across the field-aware index and database.

    Parameters
    ----------
    index : FieldIndex | None
        The loaded field-aware index.
    storage : IndexStorage | None
        For saving/loading the index.  Uses ``settings.INDEX_PATH``
        if not provided.
    """

    def __init__(
        self,
        index: FieldIndex | None = None,
        storage: IndexStorage | None = None,
    ) -> None:
        self._index = index or FieldIndex()
        self._storage = storage or IndexStorage(settings.INDEX_PATH)
        self._engine = SearchEngine(self._index)
        self._snippet_gen = SnippetGenerator()

    @property
    def index(self) -> FieldIndex:
        return self._index

    @property
    def engine(self) -> SearchEngine:
        return self._engine

    # ── Search ───────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        page: int = 1,
        limit: int = 10,
        db: AsyncSession,
    ) -> SearchResponse:
        """
        Execute a full search: parse → index → rank → DB → response.

        Parameters
        ----------
        query : str
            Raw user query.
        page : int
            1-based page number.
        limit : int
            Results per page.
        db : AsyncSession
            Database session for document retrieval.

        Returns
        -------
        SearchResponse
            Pydantic model ready for API serialisation.
        """
        start = time.perf_counter()

        offset = (page - 1) * limit

        # 1. Search engine: parse → candidate retrieval → field-weighted BM25 → paginate
        result = self._engine.search(query, limit=limit, offset=offset)

        # 2. Fetch documents from SQLite in ranked order
        ranked_ids = [r.document_id for r in result.ranked]
        repo = DocumentRepository(db)
        docs = await repo.get_by_ids(ranked_ids)

        # Build a map for fast lookup (preserving search order)
        doc_map = {doc.id: doc for doc in docs}

        # 3. Build response items with snippets (skip stale index references)
        items: list[SearchResultItem] = []
        for candidate in result.ranked:
            doc = doc_map.get(candidate.document_id)
            if doc is None:
                # Index references a deleted document — skip silently
                continue

            # Generate snippet for this result (current page only).
            snippet_obj = self._snippet_gen.generate(
                title=doc.title,
                body=doc.text,
                query=result.parsed_query or result.terms,
            )
            snippet_html = snippet_obj.to_html() if snippet_obj else None

            items.append(
                SearchResultItem(
                    document_id=candidate.document_id,
                    title=doc.title,
                    url=doc.url,
                    score=candidate.score,
                    snippet=snippet_html,
                )
            )

        elapsed = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=query,
            total=result.total,
            page=page,
            limit=limit,
            results=items,
            took_ms=round(elapsed, 2),
        )

    # ── Index management ─────────────────────────────────────

    def load_index(self) -> None:
        """
        Load the persisted index from disk.

        If the index file does not exist or is outdated (pre-field-aware),
        an empty index is used.  Never crashes the application.
        """
        try:
            if self._storage.exists():
                self._index = self._storage.load()
                self._engine = SearchEngine(self._index)
                logger.info(
                    "Index loaded: %d terms, %d documents",
                    self._index.vocabulary_size,
                    self._index.get_total_documents(),
                )
            else:
                logger.info("No persisted index found — starting with empty index")
        except IndexStorageError as exc:
            logger.error("Failed to load index: %s — using empty index", exc)
            self._index = FieldIndex()
            self._engine = SearchEngine(self._index)

    async def rebuild_index(self, db: AsyncSession) -> None:
        """
        Rebuild the entire field-aware index from SQLite documents.

        1. Load all documents from the database.
        2. Tokenise title and body for each one.
        3. Persist the new index to disk.
        """
        repo = DocumentRepository(db)
        documents = await repo.get_all_documents()

        new_index = FieldIndex()
        indexer = DocumentIndexer(new_index)

        for doc in documents:
            indexer.index_document(
                doc.id, doc.text, title=doc.title or ""
            )

        self._storage.save(new_index)
        self._index = new_index
        self._engine = SearchEngine(new_index)

        logger.info(
            "Index rebuilt: %d documents, %d terms",
            new_index.get_total_documents(),
            new_index.vocabulary_size,
        )


# ── Singleton ────────────────────────────────────────────────

_search_service: SearchService | None = None


def get_search_service() -> SearchService:
    """Return the application-wide search service singleton."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
        _search_service.load_index()
    return _search_service


def set_search_service(service: SearchService) -> None:
    """Override the singleton (used by tests)."""
    global _search_service
    _search_service = service
