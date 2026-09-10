"""
Search engine — query → rank → results.

Implements BM25 ranking (Phase 7).

Ranking criteria:

1. **BM25 relevance score** — descending
2. **Document ID** — ascending (deterministic tie-breaker)

The search pipeline:

1. Parse query (Phase 3 tokenizer)
2. Retrieve candidates (Phase 4 inverted index, OR semantics)
3. Rank by BM25 (Phase 7)
4. Paginate
"""

from __future__ import annotations

import time

from app.config.settings import settings
from app.indexing.inverted_index import InvertedIndex
from app.search.bm25_ranker import RankedCandidate
from app.search.bm25_ranker import rank_candidates as bm25_rank
from app.search.candidate_retrieval import retrieve_candidates
from app.search.query_parser import parse_query
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SearchEngine:
    """
    Coordinates query parsing, candidate retrieval, and BM25 ranking.

    Independent of FastAPI and SQLAlchemy — a pure search component.

    Parameters
    ----------
    index : InvertedIndex
        The inverted index to search against.
    k1 : float | None
        BM25 term-frequency saturation.  Defaults to ``settings.BM25_K1``.
    b : float | None
        BM25 document-length normalization.  Defaults to ``settings.BM25_B``.
    """

    def __init__(
        self,
        index: InvertedIndex,
        *,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        self._index = index
        self._k1 = k1 if k1 is not None else settings.BM25_K1
        self._b = b if b is not None else settings.BM25_B

    @property
    def index(self) -> InvertedIndex:
        return self._index

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """
        Execute a search query and return ranked results.

        Parameters
        ----------
        query : str
            Raw user query.
        limit : int
            Maximum results to return.
        offset : int
            Number of results to skip (for pagination).

        Returns
        -------
        SearchResult
            Ranked candidates with pagination metadata.
        """
        start = time.perf_counter()

        # 1. Parse query
        terms = parse_query(query)

        if not terms:
            elapsed = (time.perf_counter() - start) * 1000
            return SearchResult(
                query=query,
                terms=terms,
                ranked=[],
                total=0,
                took_ms=round(elapsed, 2),
            )

        # 2. Retrieve candidates via inverted index (OR semantics)
        candidates = retrieve_candidates(self._index, terms)

        # 3. Rank by BM25
        candidate_ids = [c.document_id for c in candidates]
        ranked = bm25_rank(
            terms, candidate_ids, self._index,
            k1=self._k1, b=self._b,
        )

        total = len(ranked)

        # 4. Paginate
        page_ranked = ranked[offset : offset + limit]

        elapsed = (time.perf_counter() - start) * 1000

        logger.debug(
            "Search '%s': %d terms, %d candidates, %.1f ms",
            query,
            len(terms),
            total,
            elapsed,
        )

        return SearchResult(
            query=query,
            terms=terms,
            ranked=page_ranked,
            total=total,
            took_ms=round(elapsed, 2),
        )


class SearchResult:
    """Encapsulates the result of a search engine query."""

    __slots__ = ("query", "terms", "ranked", "total", "took_ms")

    def __init__(
        self,
        *,
        query: str,
        terms: list[str],
        ranked: list[RankedCandidate],
        total: int,
        took_ms: float,
    ) -> None:
        self.query = query
        self.terms = terms
        self.ranked = ranked
        self.total = total
        self.took_ms = took_ms
