"""
Score normalization — transforms raw scores to a common [0, 1] scale.

Uses min-max normalization so that BM25 and cosine similarity scores
become directly comparable before fusion.

Edge-case policy (documented and tested):
- If all scores are equal → all normalized to 1.0
- Empty input → empty output
"""

from __future__ import annotations


def min_max_normalize(scores: list[float]) -> list[float]:
    """
    Min-max normalize a list of scores to [0, 1].

    Parameters
    ----------
    scores : list[float]
        Raw scores to normalize.

    Returns
    -------
    list[float]
        Normalized scores in the same order.

    Edge Cases
    ----------
    - Empty list → empty list
    - All equal → all 1.0
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)
    span = max_s - min_s

    if span == 0.0:
        # All scores identical → normalize to 1.0.
        return [1.0] * len(scores)

    return [(s - min_s) / span for s in scores]


def normalize_score_dict(
    scores: dict[int, float],
) -> dict[int, float]:
    """
    Min-max normalize scores keyed by document ID.

    Parameters
    ----------
    scores : dict[int, float]
        {document_id: raw_score}

    Returns
    -------
    dict[int, float]
        {document_id: normalized_score}
    """
    if not scores:
        return {}

    doc_ids = list(scores.keys())
    raw = [scores[did] for did in doc_ids]
    normed = min_max_normalize(raw)

    return dict(zip(doc_ids, normed))
