"""Tests for query term weighting."""

import pytest
from app.search.query_weighting import query_term_weight

def test_query_term_weight_single():
    assert query_term_weight(1) == 1.0

def test_query_term_weight_increases_sublinearly():
    w1 = query_term_weight(1)
    w2 = query_term_weight(2)
    w3 = query_term_weight(3)
    
    assert w2 > w1
    assert w3 > w2
    # Should be sublinear: diff between 3 and 2 < diff between 2 and 1
    assert (w3 - w2) < (w2 - w1)

def test_query_term_weight_cap():
    # frequency=100 would normally be 1 + ln(100) ~ 5.6
    # With cap=3.0, it should be 3.0
    w_large = query_term_weight(100, max_weight=3.0)
    assert w_large == 3.0

def test_query_term_weight_invalid():
    with pytest.raises(ValueError):
        query_term_weight(0)
    
    with pytest.raises(ValueError):
        query_term_weight(-1)


def test_query_term_weight_frequency_10():
    import math
    # 1 + ln(10) ≈ 3.30, which exceeds default cap of 3.0
    w = query_term_weight(10)
    assert w == 3.0  # capped
    # Without cap, it would be higher
    uncapped = query_term_weight(10, max_weight=10.0)
    expected = 1 + math.log(10)
    assert abs(uncapped - expected) < 1e-9

def test_query_term_weight_frequency_100_capped():
    w = query_term_weight(100)
    assert w == 3.0

def test_query_term_weight_never_negative():
    for freq in [1, 2, 3, 5, 10, 50, 100]:
        assert query_term_weight(freq) >= 0.0

def test_query_term_weight_monotonic():
    """Weight should increase (or stay at cap) as frequency increases."""
    prev = 0.0
    for freq in [1, 2, 3, 5, 10, 20, 50, 100]:
        w = query_term_weight(freq)
        assert w >= prev
        prev = w
