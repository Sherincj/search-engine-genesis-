"""
Candidate merger — creates the union of BM25 and semantic candidates.

Missing-score policy (documented and tested):
- If a document appears only in BM25 results, its semantic score is 0.0.
- If a document appears only in semantic results, its BM25 score is 0.0.
- This is applied BEFORE normalization.
"""

from __future__ import annotations

from app.hybrid_search.types import HybridCandidate


def merge_candidates(
    lexical_scores: dict[int, float],
    semantic_scores: dict[int, float],
) -> list[HybridCandidate]:
    """
    Merge BM25 and semantic results into a unified candidate set.

    Parameters
    ----------
    lexical_scores : dict[int, float]
        {document_id: bm25_score} from lexical retrieval.
    semantic_scores : dict[int, float]
        {document_id: cosine_similarity} from semantic retrieval.

    Returns
    -------
    list[HybridCandidate]
        Union of all candidates with scores (0.0 for missing).
    """
    all_ids = set(lexical_scores.keys()) | set(semantic_scores.keys())

    candidates = []
    for doc_id in all_ids:
        candidates.append(HybridCandidate(
            document_id=doc_id,
            lexical_score=lexical_scores.get(doc_id, 0.0),
            semantic_score=semantic_scores.get(doc_id, 0.0),
        ))

    return candidates
