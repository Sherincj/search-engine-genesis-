"""
Hybrid search types — result structures for score fusion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridCandidate:
    """A merged candidate from lexical and semantic retrieval."""
    document_id: int
    lexical_score: float       # raw BM25 score (0 if absent)
    semantic_score: float      # raw cosine similarity (0 if absent)


@dataclass(frozen=True)
class HybridScore:
    """A fully scored hybrid result with debug information."""
    document_id: int
    lexical_score: float
    semantic_score: float
    normalized_lexical: float
    normalized_semantic: float
    hybrid_score: float
