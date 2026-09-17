"""
Search domain types — internal models for the search pipeline.

These types flow through the search pipeline but are NOT exposed
directly as API response models.  Pydantic API schemas live in
``app/models/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateMatch:
    """
    A document identified as matching a query via the inverted index.

    Populated by candidate retrieval before database lookup.
    """

    document_id: int
    matched_terms: set[str] = field(default_factory=set)
    total_term_frequency: int = 0

    @property
    def matched_term_count(self) -> int:
        """Number of unique query terms matched."""
        return len(self.matched_terms)


@dataclass
class SearchHit:
    """
    A scored search result after ranking and database enrichment.

    Ready to be serialised into the API response.
    """

    document_id: int
    title: str
    url: str
    score: float
    snippet: str | None = None
