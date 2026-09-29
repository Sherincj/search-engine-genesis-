"""Tests for evaluation metrics — hand-calculated examples."""

import math
import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    reciprocal_rank,
    average_precision,
    mean_average_precision,
    dcg_at_k,
    ndcg_at_k,
)


# ── Precision@K ──────────────────────────────────────────

def test_precision_at_k_basic():
    # [A, B, C, D, E], relevant={A, C}
    assert precision_at_k([1, 2, 3, 4, 5], {1, 3}, 5) == pytest.approx(0.4)

def test_precision_at_k_perfect():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)

def test_precision_at_k_none_relevant():
    assert precision_at_k([1, 2, 3], {10, 20}, 3) == pytest.approx(0.0)

def test_precision_at_k_1():
    assert precision_at_k([1, 2, 3], {1}, 1) == pytest.approx(1.0)
    assert precision_at_k([2, 1, 3], {1}, 1) == pytest.approx(0.0)

def test_precision_at_k_0():
    assert precision_at_k([1, 2], {1}, 0) == 0.0

def test_precision_at_k_exceeds_results():
    # K=10 but only 3 results: denominator is still K=10
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 10) == pytest.approx(0.3)

def test_precision_at_k_empty_results():
    assert precision_at_k([], {1, 2}, 5) == 0.0


# ── Recall@K ─────────────────────────────────────────────

def test_recall_at_k_basic():
    # 4 relevant, 2 found in top-5
    assert recall_at_k([1, 2, 3, 4, 5], {1, 3, 10, 20}, 5) == pytest.approx(0.5)

def test_recall_at_k_all_found():
    assert recall_at_k([1, 2, 3], {1, 2}, 3) == pytest.approx(1.0)

def test_recall_at_k_none_found():
    assert recall_at_k([1, 2, 3], {10, 20}, 5) == pytest.approx(0.0)

def test_recall_at_k_no_relevant():
    # No relevant documents -> return 0.0 (not NaN)
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0

def test_recall_at_k_0():
    assert recall_at_k([1, 2], {1}, 0) == 0.0


# ── F1@K ─────────────────────────────────────────────────

def test_f1_at_k_basic():
    # P@3=2/3, R@3=2/2=1.0, F1 = 2*(2/3)*1/(2/3+1) = 0.8
    assert f1_at_k([1, 5, 2], {1, 2}, 3) == pytest.approx(0.8)

def test_f1_at_k_zero():
    assert f1_at_k([1, 2, 3], {10}, 3) == 0.0

def test_f1_at_k_perfect():
    assert f1_at_k([1, 2], {1, 2}, 2) == pytest.approx(1.0)


# ── Reciprocal Rank ──────────────────────────────────────

def test_rr_first():
    assert reciprocal_rank([1, 2, 3], {1}) == pytest.approx(1.0)

def test_rr_third():
    assert reciprocal_rank([5, 6, 1, 2], {1}) == pytest.approx(1/3)

def test_rr_not_found():
    assert reciprocal_rank([1, 2, 3], {10}) == 0.0

def test_rr_empty():
    assert reciprocal_rank([], {1}) == 0.0

def test_rr_multiple_relevant():
    # RR uses FIRST relevant
    assert reciprocal_rank([5, 1, 2], {1, 2}) == pytest.approx(0.5)


# ── Average Precision ────────────────────────────────────

def test_ap_basic():
    # Retrieved: [1, 5, 2, 3], Relevant: {1, 2, 3}
    # rank 1: P@1 = 1/1 = 1.0
    # rank 3: P@3 = 2/3
    # rank 4: P@4 = 3/4
    # AP = (1 + 2/3 + 3/4) / 3 = 2.4167 / 3 = 0.8056
    ap = average_precision([1, 5, 2, 3], {1, 2, 3})
    assert ap == pytest.approx((1 + 2/3 + 3/4) / 3)

def test_ap_perfect():
    # All relevant at top
    ap = average_precision([1, 2, 3], {1, 2, 3})
    assert ap == pytest.approx(1.0)

def test_ap_none_found():
    assert average_precision([10, 20, 30], {1, 2}) == pytest.approx(0.0)

def test_ap_no_relevant():
    assert average_precision([1, 2, 3], set()) == 0.0

def test_ap_one_relevant_at_end():
    # [5, 6, 7, 1], relevant={1}
    # Only rank 4: P@4 = 1/4
    # AP = (1/4) / 1 = 0.25
    assert average_precision([5, 6, 7, 1], {1}) == pytest.approx(0.25)


# ── MAP ──────────────────────────────────────────────────

def test_map_basic():
    q1 = ([1, 2, 3], {1, 3})  # AP = (1 + 2/3)/2 = 0.8333
    q2 = ([5, 1, 2], {1, 2})  # AP = (1/2 + 2/3)/2 = 0.5833
    m = mean_average_precision([q1, q2])
    expected = ((1 + 2/3)/2 + (1/2 + 2/3)/2) / 2
    assert m == pytest.approx(expected)

def test_map_empty():
    assert mean_average_precision([]) == 0.0

def test_map_includes_zero_ap():
    q1 = ([1, 2], {1})   # AP = 1.0
    q2 = ([3, 4], {10})  # AP = 0.0
    m = mean_average_precision([q1, q2])
    assert m == pytest.approx(0.5)


# ── DCG@K ────────────────────────────────────────────────

def test_dcg_basic():
    # Retrieved: [1, 2, 3], relevance: {1:3, 2:2, 3:1}
    # DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    #     = 7/1 + 3/1.585 + 1/2 = 7 + 1.893 + 0.5 = 9.393
    relevance = {1: 3, 2: 2, 3: 1}
    dcg = dcg_at_k([1, 2, 3], relevance, 3)
    expected = 7/math.log2(2) + 3/math.log2(3) + 1/math.log2(4)
    assert dcg == pytest.approx(expected)

def test_dcg_k_0():
    assert dcg_at_k([1, 2], {1: 3}, 0) == 0.0

def test_dcg_unknown_doc():
    # Unknown docs treated as relevance 0
    assert dcg_at_k([99], {1: 3}, 1) == pytest.approx(0.0)


# ── NDCG@K ───────────────────────────────────────────────

def test_ndcg_perfect():
    # Already in ideal order
    relevance = {1: 3, 2: 2, 3: 1}
    assert ndcg_at_k([1, 2, 3], relevance, 3) == pytest.approx(1.0)

def test_ndcg_reverse():
    # Worst possible order
    relevance = {1: 3, 2: 2, 3: 1}
    score = ndcg_at_k([3, 2, 1], relevance, 3)
    assert 0 < score < 1.0

def test_ndcg_no_relevant():
    assert ndcg_at_k([1, 2, 3], {}, 3) == 0.0

def test_ndcg_k_0():
    assert ndcg_at_k([1, 2], {1: 3}, 0) == 0.0

def test_ndcg_partial():
    # Only some relevant docs in top K
    relevance = {1: 3, 2: 2, 3: 1, 4: 0}
    score = ndcg_at_k([4, 1, 3, 2], relevance, 3)
    # Not perfect, not zero
    assert 0 < score < 1.0

def test_ndcg_k_exceeds_results():
    relevance = {1: 3, 2: 2}
    score = ndcg_at_k([1, 2], relevance, 10)
    assert score == pytest.approx(1.0)  # Perfect order, even with K>len


# ── Edge cases ───────────────────────────────────────────

def test_all_relevant():
    retrieved = [1, 2, 3]
    relevant = {1, 2, 3}
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1.0)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)
    assert f1_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

def test_one_relevant():
    retrieved = [5, 6, 1]
    relevant = {1}
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1/3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

def test_relevant_beyond_k():
    retrieved = [5, 6, 7, 1]
    relevant = {1}
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(0.0)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.0)
