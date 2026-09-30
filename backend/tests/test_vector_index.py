"""Tests for the custom vector index and cosine similarity."""

import math
import pytest

from app.vector_index.vector_index import (
    VectorIndex,
    VectorIndexError,
    cosine_similarity,
)


# ── Cosine similarity ─────────────────────────────────────────

def test_cosine_identical():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

def test_cosine_identical_non_unit():
    assert cosine_similarity([2, 3], [2, 3]) == pytest.approx(1.0)

def test_cosine_orthogonal():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

def test_cosine_opposite():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

def test_cosine_known_value():
    # cos(45°) = 1/√2 ≈ 0.7071
    assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(1/math.sqrt(2))

def test_cosine_zero_vector():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

def test_cosine_both_zero():
    assert cosine_similarity([0, 0], [0, 0]) == 0.0

def test_cosine_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1, 0], [1, 0, 0])

def test_cosine_negative_values():
    sim = cosine_similarity([-1, -1], [-1, -1])
    assert sim == pytest.approx(1.0)

def test_cosine_symmetry():
    a, b = [1, 2, 3], [4, 5, 6]
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


# ── VectorIndex construction ──────────────────────────────────

def test_index_creation():
    idx = VectorIndex(dimension=3)
    assert idx.dimension == 3
    assert idx.count() == 0

def test_index_invalid_dimension():
    with pytest.raises(VectorIndexError):
        VectorIndex(dimension=0)
    with pytest.raises(VectorIndexError):
        VectorIndex(dimension=-1)


# ── Add ───────────────────────────────────────────────────────

def test_add_and_contains():
    idx = VectorIndex(dimension=3)
    idx.add(1, [0.1, 0.2, 0.3])
    assert idx.contains(1)
    assert idx.count() == 1

def test_add_duplicate_replaces():
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 0.0])
    idx.add(1, [0.0, 1.0])
    assert idx.count() == 1
    vec = idx.get_vector(1)
    assert vec == [0.0, 1.0]

def test_add_wrong_dimension():
    idx = VectorIndex(dimension=3)
    with pytest.raises(VectorIndexError):
        idx.add(1, [1.0, 2.0])  # dimension 2 != 3

def test_add_empty_vector():
    idx = VectorIndex(dimension=3)
    with pytest.raises(VectorIndexError):
        idx.add(1, [])

def test_add_nan_rejected():
    idx = VectorIndex(dimension=2)
    with pytest.raises(VectorIndexError):
        idx.add(1, [1.0, float('nan')])

def test_add_inf_rejected():
    idx = VectorIndex(dimension=2)
    with pytest.raises(VectorIndexError):
        idx.add(1, [1.0, float('inf')])

def test_add_negative_inf_rejected():
    idx = VectorIndex(dimension=2)
    with pytest.raises(VectorIndexError):
        idx.add(1, [float('-inf'), 1.0])

def test_add_negative_id_rejected():
    idx = VectorIndex(dimension=2)
    with pytest.raises(VectorIndexError):
        idx.add(-1, [1.0, 2.0])


# ── Remove ────────────────────────────────────────────────────

def test_remove_existing():
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 0.0])
    assert idx.remove(1) is True
    assert not idx.contains(1)
    assert idx.count() == 0

def test_remove_nonexistent():
    idx = VectorIndex(dimension=2)
    assert idx.remove(999) is False


# ── Update ────────────────────────────────────────────────────

def test_update_replaces():
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 0.0])
    idx.update(1, [0.0, 1.0])
    assert idx.get_vector(1) == [0.0, 1.0]


# ── Clear ─────────────────────────────────────────────────────

def test_clear():
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 0.0])
    idx.add(2, [0.0, 1.0])
    idx.clear()
    assert idx.count() == 0


# ── Search ────────────────────────────────────────────────────

def test_search_basic():
    idx = VectorIndex(dimension=3)
    idx.add(1, [1, 0, 0])
    idx.add(2, [0, 1, 0])
    idx.add(3, [0, 0, 1])
    results = idx.search([1, 0, 0], top_k=3)
    assert len(results) == 3
    assert results[0].document_id == 1
    assert results[0].score == pytest.approx(1.0)

def test_search_top_k():
    idx = VectorIndex(dimension=2)
    for i in range(10):
        idx.add(i, [float(i), 1.0])
    results = idx.search([9, 1], top_k=3)
    assert len(results) == 3

def test_search_empty_index():
    idx = VectorIndex(dimension=2)
    results = idx.search([1, 0], top_k=5)
    assert results == []

def test_search_top_k_zero():
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 0.0])
    assert idx.search([1, 0], top_k=0) == []

def test_search_deterministic_tiebreak():
    """Equal scores: sort by document_id ASC."""
    idx = VectorIndex(dimension=2)
    idx.add(5, [1.0, 0.0])
    idx.add(2, [1.0, 0.0])
    idx.add(8, [1.0, 0.0])
    results = idx.search([1.0, 0.0], top_k=3)
    ids = [r.document_id for r in results]
    assert ids == [2, 5, 8]  # ascending by ID for equal scores

def test_search_ordering():
    """Higher similarity first."""
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 0.0])    # exact match
    idx.add(2, [0.7, 0.7])    # partial match
    idx.add(3, [0.0, 1.0])    # orthogonal
    results = idx.search([1.0, 0.0], top_k=3)
    assert results[0].document_id == 1
    assert results[0].score > results[1].score
    assert results[1].score > results[2].score

def test_search_wrong_dimension():
    idx = VectorIndex(dimension=3)
    idx.add(1, [1, 0, 0])
    with pytest.raises(VectorIndexError):
        idx.search([1, 0], top_k=1)  # dimension 2 != 3


# ── Validation ────────────────────────────────────────────────

def test_validate_healthy():
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 0.0])
    idx.add(2, [0.0, 1.0])
    idx.validate()  # should not raise

def test_get_all_ids():
    idx = VectorIndex(dimension=2)
    idx.add(3, [1.0, 0.0])
    idx.add(1, [0.0, 1.0])
    ids = sorted(idx.get_all_ids())
    assert ids == [1, 3]

def test_get_vector_not_found():
    idx = VectorIndex(dimension=2)
    assert idx.get_vector(999) is None

def test_get_vector_defensive_copy():
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 2.0])
    vec = idx.get_vector(1)
    vec[0] = 999.0  # mutate the copy
    assert idx.get_vector(1) == [1.0, 2.0]  # original unchanged
