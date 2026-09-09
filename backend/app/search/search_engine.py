"""
Search engine — query → rank → results.

Implements TF-IDF cosine similarity ranking (Phase 6).

Ranking criteria:

1. **Cosine similarity** (TF-IDF) — descending
2. **Document ID** — ascending (deterministic tie-breaker)

The search pipeline:

1. Parse query (Phase 3 tokenizer)
2. Retrieve candidates (Phase 4 inverted index, OR semantics)
3. Rank by TF-IDF cosine similarity (Phase 6)
4. Paginate
"""

from __future__ import annotations

import time

from app.indexing.inverted_index import InvertedIndex
from app.search.candidate_retrieval import retrieve_candidates
from app.search.query_parser import parse_query
from app.search.tfidf_ranker import RankedCandidate, rank_candidates
from app.search.types import CandidateMatch
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SearchEngine:
    """
    Coordinates query parsing, candidate retrieval, and TF-IDF ranking.

    Independent of FastAPI and SQLAlchemy — a pure search component.

    Parameters
    ----------
    index : InvertedIndex
        The inverted index to search against.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index

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

        # 3. Rank by TF-IDF cosine similarity
        candidate_ids = [c.document_id for c in candidates]
        ranked = rank_candidates(terms, candidate_ids, self._index)

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
