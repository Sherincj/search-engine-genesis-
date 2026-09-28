"""
Query Processor — transforms raw queries into a processed, weighted representation.

Phase 12 update: handles term frequencies, sublinear weighting, and deduplicates phrases.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.search.query_parser import QueryPhrase, parse_query_structured
from app.search.query_weighting import query_term_weight
from app.text_processing import tokenize


@dataclass(frozen=True)
class QueryTerm:
    """A single query term with its frequency and assigned weight."""
    term: str
    frequency: int
    weight: float


@dataclass(frozen=True)
class ProcessedQuery:
    """
    Fully processed query with term weights, unique phrases, and metadata.
    """
    terms: tuple[QueryTerm, ...]       # All unique terms with global frequency
    phrases: tuple[QueryPhrase, ...]
    original_query: str
    loose_terms: tuple[str, ...]       # Terms that appear outside of quotes

    @property
    def is_empty(self) -> bool:
        return len(self.terms) == 0 and len(self.phrases) == 0

    @property
    def has_phrases(self) -> bool:
        return len(self.phrases) > 0
        
    @property
    def is_phrase_only(self) -> bool:
        return len(self.loose_terms) == 0 and self.has_phrases

    @property
    def all_terms(self) -> tuple[str, ...]:
        """All unique terms for candidate retrieval, in deterministic order."""
        return tuple(qt.term for qt in self.terms)


class QueryProcessor:
    """
    Processes raw text queries into ProcessedQuery objects.
    Applies frequency counting and term weighting.
    """

    def __init__(self, max_weight: float = 3.0) -> None:
        self.max_weight = max_weight

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a raw search query.

        - Parses phrases using existing query parser.
        - Tokenizes the full query to count term frequencies (preserving first-appearance order).
        - Deduplicates identical phrases.
        - Calculates sublinear term weights.
        """
        if not query or not query.strip():
            return ProcessedQuery(terms=(), phrases=(), original_query=query, loose_terms=())

        parsed = parse_query_structured(query)

        # 1. Deduplicate phrases (preserve first appearance)
        unique_phrases: list[QueryPhrase] = []
        seen_phrases: set[tuple[str, ...]] = set()
        
        for p in parsed.phrases:
            if p.terms not in seen_phrases:
                seen_phrases.add(p.terms)
                unique_phrases.append(p)

        # 2. Count term frequencies across ALL tokens in the query
        # Using tokenize(query) directly gives us the true first-appearance order
        # and frequency of all terms, regardless of whether they are quoted.
        flat_terms = tokenize(query)
        
        term_counts: dict[str, int] = {}
        ordered_unique_terms: list[str] = []
        
        for t in flat_terms:
            if t not in term_counts:
                term_counts[t] = 0
                ordered_unique_terms.append(t)
            term_counts[t] += 1

        # 3. Create weighted QueryTerm objects
        query_terms: list[QueryTerm] = []
        for t in ordered_unique_terms:
            freq = term_counts[t]
            weight = query_term_weight(freq, self.max_weight)
            query_terms.append(QueryTerm(term=t, frequency=freq, weight=weight))

        return ProcessedQuery(
            terms=tuple(query_terms),
            phrases=tuple(unique_phrases),
            original_query=query,
            loose_terms=parsed.terms,
        )
