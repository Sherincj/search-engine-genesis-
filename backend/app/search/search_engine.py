"""
Search engine — query → rank → results.

Phase 11 update: phrase-aware search pipeline::

    1. Parse query → terms + phrases
    2. Retrieve candidates (OR semantics for all terms)
    3. For phrase-only queries, filter to phrase matches
    4. Rank by field-weighted BM25
    5. Add phrase boost for phrase matches
    6. Paginate

Ranking criteria:

1. **Field-weighted BM25 score + phrase boost** — descending
2. **Document ID** — ascending (deterministic tie-breaker)
"""

from __future__ import annotations

import time

from app.config.settings import settings
from app.indexing.field_index import FieldIndex
from app.search.bm25_ranker import RankedCandidate
from app.search.bm25_ranker import rank_candidates as bm25_rank
from app.search.candidate_retrieval import retrieve_candidates
from app.search.phrase_matching import find_phrase_matches, get_phrase_candidates
from app.search.query_parser import parse_query, parse_query_structured, ParsedQuery
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SearchEngine:
    """
    Coordinates query parsing, candidate retrieval, phrase matching,
    and field-weighted BM25 ranking.

    Independent of FastAPI and SQLAlchemy — a pure search component.

    Parameters
    ----------
    index : FieldIndex
        The field-aware inverted index to search against.
    k1 : float | None
        BM25 term-frequency saturation.  Defaults to ``settings.BM25_K1``.
    b : float | None
        BM25 document-length normalization.  Defaults to ``settings.BM25_B``.
    title_weight : float | None
        Title field weight.  Defaults to ``settings.BM25_TITLE_WEIGHT``.
    body_weight : float | None
        Body field weight.  Defaults to ``settings.BM25_BODY_WEIGHT``.
    phrase_boost : float | None
        Score boost per phrase match.  Defaults to ``settings.PHRASE_BOOST``.
    """

    def __init__(
        self,
        index: FieldIndex,
        *,
        k1: float | None = None,
        b: float | None = None,
        title_weight: float | None = None,
        body_weight: float | None = None,
        phrase_boost: float | None = None,
    ) -> None:
        self._index = index
        self._k1 = k1 if k1 is not None else settings.BM25_K1
        self._b = b if b is not None else settings.BM25_B
        self._title_weight = (
            title_weight if title_weight is not None
            else settings.BM25_TITLE_WEIGHT
        )
        self._body_weight = (
            body_weight if body_weight is not None
            else settings.BM25_BODY_WEIGHT
        )
        self._phrase_boost = (
            phrase_boost if phrase_boost is not None
            else settings.PHRASE_BOOST
        )

    @property
    def index(self) -> FieldIndex:
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
            Raw user query (may contain quoted phrases).
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

        # 1. Parse query into structured representation.
        parsed = parse_query_structured(query)
        terms = list(parsed.all_terms)

        if parsed.is_empty:
            elapsed = (time.perf_counter() - start) * 1000
            return SearchResult(
                query=query,
                terms=terms,
                parsed_query=parsed,
                ranked=[],
                total=0,
                took_ms=round(elapsed, 2),
            )

        # 2. Retrieve candidates via field index (OR semantics).
        candidates = retrieve_candidates(self._index, terms)

        # 3. For phrase-only queries, filter to phrase matches.
        if parsed.has_phrases:
            # Build set of doc IDs that match ALL phrases.
            phrase_matching_docs: set[int] | None = None
            phrase_doc_matches: dict[int, int] = {}  # doc_id → phrase match count

            for phrase in parsed.phrases:
                # Get candidates that have all phrase terms.
                phrase_candidates = get_phrase_candidates(
                    list(phrase.terms), self._index
                )

                # Verify actual phrase matches via positions.
                verified: set[int] = set()
                for doc_id in phrase_candidates:
                    matches = find_phrase_matches(
                        list(phrase.terms), doc_id, self._index
                    )
                    if matches:
                        verified.add(doc_id)
                        phrase_doc_matches[doc_id] = (
                            phrase_doc_matches.get(doc_id, 0) + 1
                        )

                if phrase_matching_docs is None:
                    phrase_matching_docs = verified
                else:
                    phrase_matching_docs &= verified

            # If query is ONLY phrases (no loose terms), filter strictly.
            if not parsed.terms and phrase_matching_docs is not None:
                candidates = [
                    c for c in candidates
                    if c.document_id in phrase_matching_docs
                ]

        # 4. Rank by field-weighted BM25.
        candidate_ids = [c.document_id for c in candidates]
        ranked = bm25_rank(
            terms, candidate_ids, self._index,
            k1=self._k1, b=self._b,
            title_weight=self._title_weight,
            body_weight=self._body_weight,
        )

        # 5. Apply phrase boost.
        if parsed.has_phrases and phrase_doc_matches:
            for r in ranked:
                match_count = phrase_doc_matches.get(r.document_id, 0)
                if match_count > 0:
                    # Apply phrase boost per matched phrase, weighted by field.
                    r.score += self._phrase_boost * match_count

            # Re-sort after boosting.
            ranked.sort(key=lambda r: (-r.score, r.document_id))

        total = len(ranked)

        # 6. Paginate.
        page_ranked = ranked[offset : offset + limit]

        elapsed = (time.perf_counter() - start) * 1000

        logger.debug(
            "Search '%s': %d terms, %d phrases, %d candidates, %.1f ms",
            query,
            len(terms),
            len(parsed.phrases),
            total,
            elapsed,
        )

        return SearchResult(
            query=query,
            terms=terms,
            parsed_query=parsed,
            ranked=page_ranked,
            total=total,
            took_ms=round(elapsed, 2),
        )


class SearchResult:
    """Encapsulates the result of a search engine query."""

    __slots__ = ("query", "terms", "parsed_query", "ranked", "total", "took_ms")

    def __init__(
        self,
        *,
        query: str,
        terms: list[str],
        parsed_query: ParsedQuery | None = None,
        ranked: list[RankedCandidate],
        total: int,
        took_ms: float,
    ) -> None:
        self.query = query
        self.terms = terms
        self.parsed_query = parsed_query
        self.ranked = ranked
        self.total = total
        self.took_ms = took_ms
