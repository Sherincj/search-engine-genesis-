"""Tests for min-max score normalization."""

import pytest

from app.hybrid_search.normalization import min_max_normalize, normalize_score_dict


# ── min_max_normalize ─────────────────────────────────────────

def test_normalize_basic():
    result = min_max_normalize([2.0, 5.0, 10.0])
    assert result == pytest.approx([0.0, 0.375, 1.0])


def test_normalize_single():
    result = min_max_normalize([5.0])
    assert result == [1.0]


def test_normalize_all_equal():
    result = min_max_normalize([3.0, 3.0, 3.0])
    assert result == [1.0, 1.0, 1.0]


def test_normalize_two_values():
    result = min_max_normalize([0.0, 1.0])
    assert result == pytest.approx([0.0, 1.0])


def test_normalize_negative_scores():
    result = min_max_normalize([-10.0, 0.0, 10.0])
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_zero_scores():
    result = min_max_normalize([0.0, 0.0, 0.0])
    assert result == [1.0, 1.0, 1.0]


def test_normalize_empty():
    assert min_max_normalize([]) == []


def test_normalize_preserves_order():
    result = min_max_normalize([1.0, 3.0, 2.0])
    assert result[1] > result[2] > result[0]


def test_normalize_already_01():
    result = min_max_normalize([0.0, 0.5, 1.0])
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_large_range():
    result = min_max_normalize([0.0, 100.0])
    assert result == pytest.approx([0.0, 1.0])


def test_normalize_cosine_range():
    """Cosine similarity values in [-1, 1]."""
    result = min_max_normalize([-0.5, 0.0, 0.8])
    assert result[0] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)


def test_normalize_deterministic():
    scores = [3.0, 1.0, 4.0, 1.0, 5.0]
    assert min_max_normalize(scores) == min_max_normalize(scores)


# ── normalize_score_dict ──────────────────────────────────────

def test_normalize_dict_basic():
    result = normalize_score_dict({1: 0.0, 2: 5.0, 3: 10.0})
    assert result[1] == pytest.approx(0.0)
    assert result[3] == pytest.approx(1.0)


def test_normalize_dict_empty():
    assert normalize_score_dict({}) == {}


def test_normalize_dict_single():
    result = normalize_score_dict({42: 7.5})
    assert result[42] == 1.0


def test_normalize_dict_preserves_keys():
    result = normalize_score_dict({10: 1.0, 20: 2.0, 30: 3.0})
    assert set(result.keys()) == {10, 20, 30}
