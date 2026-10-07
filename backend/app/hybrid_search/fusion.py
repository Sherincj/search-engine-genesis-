"""
Score fusion — combines normalized lexical and semantic scores.

Provides a FusionStrategy protocol for future extensibility,
with WeightedLinearFusion as the initial implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.hybrid_search.normalization import normalize_score_dict
from app.hybrid_search.types import HybridCandidate, HybridScore


@runtime_checkable
class FusionStrategy(Protocol):
    """Protocol for score fusion strategies."""

    def fuse(
        self,
        candidates: list[HybridCandidate],
    ) -> list[HybridScore]:
        """Fuse candidate scores into hybrid scores."""
        ...


class WeightedLinearFusion:
    """
    Weighted linear combination of normalized scores.

    hybrid = lexical_weight × norm_bm25 + semantic_weight × norm_semantic

    Parameters
    ----------
    lexical_weight : float
        Weight for the lexical (BM25) component.
    semantic_weight : float
        Weight for the semantic component.
    """

    def __init__(
        self,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        if lexical_weight < 0 or semantic_weight < 0:
            raise ValueError(
                f"Weights must be non-negative: "
                f"lexical={lexical_weight}, semantic={semantic_weight}"
            )
        self._lexical_weight = lexical_weight
        self._semantic_weight = semantic_weight

    @property
    def lexical_weight(self) -> float:
        return self._lexical_weight

    @property
    def semantic_weight(self) -> float:
        return self._semantic_weight

    def fuse(
        self,
        candidates: list[HybridCandidate],
    ) -> list[HybridScore]:
        """
        Normalize and fuse candidate scores.

        Returns results sorted by hybrid_score DESC, document_id ASC.
        """
        if not candidates:
            return []

        # Build score dicts for normalization.
        lex_scores = {c.document_id: c.lexical_score for c in candidates}
        sem_scores = {c.document_id: c.semantic_score for c in candidates}

        norm_lex = normalize_score_dict(lex_scores)
        norm_sem = normalize_score_dict(sem_scores)

        results = []
        for c in candidates:
            nl = norm_lex[c.document_id]
            ns = norm_sem[c.document_id]
            hybrid = self._lexical_weight * nl + self._semantic_weight * ns

            results.append(HybridScore(
                document_id=c.document_id,
                lexical_score=c.lexical_score,
                semantic_score=c.semantic_score,
                normalized_lexical=nl,
                normalized_semantic=ns,
                hybrid_score=hybrid,
            ))

        # Sort: hybrid_score DESC, document_id ASC for determinism.
        results.sort(key=lambda r: (-r.hybrid_score, r.document_id))
        return results
