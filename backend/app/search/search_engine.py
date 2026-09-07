"""
Search engine — query → rank → results.

Implements the baseline keyword ranking for Phase 5.

Ranking criteria (in order of priority):

1. **Matched unique term count** (more unique matches = better)
2. **Total term frequency** (higher TF across matches = better)
3. **Document ID ascending** (deterministic tie-breaker)

This is intentionally simple.  TF-IDF (Phase 6) and BM25 (Phase 7)
will replace the scoring logic without changing the overall pipeline.
"""

from __future__ import annotations

import time

from app.indexing.inverted_index import InvertedIndex
from app.search.candidate_retrieval import retrieve_candidates
from app.search.query_parser import parse_query
from app.search.types import CandidateMatch, SearchHit
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SearchEngine:
    """
    Coordinates query parsing, candidate retrieval, and ranking.

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
            Ranked candidate IDs with pagination metadata.
        """
        start = time.perf_counter()

        # 1. Parse query
        terms = parse_query(query)

        if not terms:
            elapsed = (time.perf_counter() - start) * 1000
            return SearchResult(
                query=query,
                terms=terms,
                candidates=[],
                total=0,
                took_ms=round(elapsed, 2),
            )

        # 2. Retrieve candidates via inverted index
        candidates = retrieve_candidates(self._index, terms)

        # 3. Rank candidates
        ranked = self._rank(candidates)

        total = len(ranked)

        # 4. Paginate
        page_candidates = ranked[offset : offset + limit]

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
            candidates=page_candidates,
            total=total,
            took_ms=round(elapsed, 2),
        )

    @staticmethod
    def _rank(candidates: list[CandidateMatch]) -> list[CandidateMatch]:
        """
        Sort candidates by the baseline ranking criteria.

        1. Matched unique term count — descending
        2. Total term frequency — descending
        3. Document ID — ascending (tie-breaker)
        """
        return sorted(
            candidates,
            key=lambda c: (
                -c.matched_term_count,
                -c.total_term_frequency,
                c.document_id,
            ),
        )


class SearchResult:
    """Encapsulates the result of a search engine query."""

    __slots__ = ("query", "terms", "candidates", "total", "took_ms")

    def __init__(
        self,
        *,
        query: str,
        terms: list[str],
        candidates: list[CandidateMatch],
        total: int,
        took_ms: float,
    ) -> None:
        self.query = query
        self.terms = terms
        self.candidates = candidates
        self.total = total
        self.took_ms = took_ms
