"""
Search evaluation metrics — pure mathematical functions.

All functions operate on ranked result lists and relevance judgments.
No dependency on BM25, SearchEngine, or any search-specific code.
"""

from __future__ import annotations

import math


def precision_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.
    
    Binary relevance: a document is relevant if it appears in `relevant`.
    
    Returns 0.0 if k <= 0.
    If k > len(retrieved), evaluates available results.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """
    Recall@K: fraction of relevant documents found in top-K.
    
    Returns 0.0 if there are no relevant documents (avoids division by zero).
    """
    if k <= 0 or not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def f1_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """
    F1@K: harmonic mean of Precision@K and Recall@K.
    
    Returns 0.0 if both precision and recall are zero.
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(
    retrieved: list[int],
    relevant: set[int],
) -> float:
    """
    Reciprocal Rank: 1/position of the first relevant result.
    
    Binary relevance: relevant if relevance > 0.
    Returns 0.0 if no relevant document is found.
    """
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def average_precision(
    retrieved: list[int],
    relevant: set[int],
) -> float:
    """
    Average Precision: mean of Precision@k at each relevant rank.
    
    For each position k where a relevant document appears,
    compute Precision@k. Then average over the total number
    of relevant documents.
    
    Returns 0.0 if there are no relevant documents.
    """
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def mean_average_precision(
    queries: list[tuple[list[int], set[int]]],
) -> float:
    """
    Mean Average Precision across multiple queries.
    
    Each element is (retrieved_ids, relevant_ids).
    Zero-result queries contribute AP=0.
    Returns 0.0 if no queries provided.
    """
    if not queries:
        return 0.0
    total = sum(average_precision(r, rel) for r, rel in queries)
    return total / len(queries)


def dcg_at_k(
    retrieved: list[int],
    relevance: dict[int, int],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain at K.
    
    DCG@K = sum((2^rel_i - 1) / log2(i + 1)) for i = 1..K
    
    `relevance` maps doc_id -> graded relevance (0-3).
    Documents not in `relevance` are treated as relevance 0.
    """
    if k <= 0:
        return 0.0
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevance.get(doc_id, 0)
        score += (2 ** rel - 1) / math.log2(i + 1)
    return score


def ndcg_at_k(
    retrieved: list[int],
    relevance: dict[int, int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    
    NDCG@K = DCG@K / IDCG@K
    
    IDCG is calculated from the ideal ordering (sorted by relevance desc).
    Returns 0.0 if IDCG is zero (no relevant documents).
    """
    if k <= 0:
        return 0.0
    actual_dcg = dcg_at_k(retrieved, relevance, k)
    
    # Build ideal ranking: sort by relevance descending.
    ideal_order = sorted(relevance.keys(), key=lambda d: relevance[d], reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, relevance, k)
    
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg
