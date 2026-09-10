"""
Tests for BM25 mathematical functions.

Covers:
- BM25 IDF (smoothed)
- BM25 term score (saturation + length normalization)
- BM25 complete score
- k1 parameter effects
- b parameter effects
- TF saturation demonstration
- Document-length normalization
- Input validation
- Hand-calculated values
- BM25 vs TF-IDF comparison
"""

import math

import pytest

from app.search.bm25 import bm25_idf, bm25_score, bm25_term_score
from app.search.tfidf import inverse_document_frequency as tfidf_idf


# ══════════════════════════════════════════════════════════════
# BM25 IDF
# ══════════════════════════════════════════════════════════════

def test_idf_rare_term():
    """N=100, DF=1 → high IDF."""
    idf = bm25_idf(100, 1)
    assert idf > 0
    expected = math.log(1 + (100 - 1 + 0.5) / (1 + 0.5))
    assert math.isclose(idf, expected)


def test_idf_common_term():
    """N=100, DF=50 → lower IDF."""
    idf = bm25_idf(100, 50)
    assert idf > 0
    assert bm25_idf(100, 1) > idf


def test_idf_all_documents_still_positive():
    """N=100, DF=100 → BM25 IDF > 0 (unlike TF-IDF where IDF=0)."""
    idf = bm25_idf(100, 100)
    assert idf > 0  # Key difference from Phase 6


def test_idf_all_documents_exact():
    """N=100, DF=100 → log(1 + 0.5/100.5)."""
    idf = bm25_idf(100, 100)
    expected = math.log(1 + (0.5) / (100.5))
    assert math.isclose(idf, expected)


def test_idf_rare_greater_than_common():
    assert bm25_idf(100, 1) > bm25_idf(100, 50) > bm25_idf(100, 100)


def test_idf_single_document():
    """N=1, DF=1 → positive but small."""
    idf = bm25_idf(1, 1)
    assert idf > 0


def test_idf_hand_calculated():
    """
    N=4, DF=1
    IDF = log(1 + (4-1+0.5)/(1+0.5)) = log(1 + 3.5/1.5) = log(1 + 2.333...)
    """
    idf = bm25_idf(4, 1)
    expected = math.log(1 + 3.5 / 1.5)
    assert math.isclose(idf, expected)


# ── IDF Validation ───────────────────────────────────────────

def test_idf_invalid_n_zero():
    with pytest.raises(ValueError, match="total_documents must be >= 1"):
        bm25_idf(0, 1)


def test_idf_invalid_df_zero():
    with pytest.raises(ValueError, match="document_frequency must be >= 1"):
        bm25_idf(100, 0)


def test_idf_invalid_df_exceeds_n():
    with pytest.raises(ValueError, match="cannot exceed"):
        bm25_idf(10, 20)


def test_idf_invalid_negative_n():
    with pytest.raises(ValueError):
        bm25_idf(-5, 1)


def test_idf_invalid_negative_df():
    with pytest.raises(ValueError):
        bm25_idf(100, -3)


# ── BM25 IDF vs TF-IDF IDF ──────────────────────────────────

def test_bm25_idf_differs_from_tfidf_idf():
    """The two IDF formulas must produce different values."""
    bm25 = bm25_idf(100, 10)
    tfidf = tfidf_idf(100, 10)
    assert not math.isclose(bm25, tfidf)


def test_bm25_idf_nonzero_when_tfidf_zero():
    """BM25 IDF > 0 when DF=N, whereas TF-IDF IDF = 0."""
    assert bm25_idf(100, 100) > 0
    assert tfidf_idf(100, 100) == 0.0


# ══════════════════════════════════════════════════════════════
# BM25 Term Score
# ══════════════════════════════════════════════════════════════

def test_term_score_basic():
    idf = bm25_idf(100, 10)
    score = bm25_term_score(
        term_frequency=1, document_length=10,
        average_document_length=10.0, idf=idf,
    )
    assert score > 0


def test_term_score_zero_tf():
    """TF=0 → contribution is 0."""
    score = bm25_term_score(
        term_frequency=0, document_length=10,
        average_document_length=10.0, idf=5.0,
    )
    assert score == 0.0


# ── TF Saturation ────────────────────────────────────────────

def test_tf_saturation():
    """Increasing TF increases score but with diminishing returns."""
    idf = bm25_idf(100, 10)
    scores = []
    for tf in [1, 2, 10, 100]:
        s = bm25_term_score(
            term_frequency=tf, document_length=10,
            average_document_length=10.0, idf=idf,
        )
        scores.append(s)

    # Scores increase
    assert scores[1] > scores[0]
    assert scores[2] > scores[1]
    assert scores[3] > scores[2]

    # Diminishing returns: gap shrinks
    gap1 = scores[1] - scores[0]   # TF 1→2
    gap2 = scores[2] - scores[1]   # TF 2→10
    gap3 = scores[3] - scores[2]   # TF 10→100

    # Each per-unit gap shrinks
    assert gap2 / 8 < gap1  # 8 units of TF gained
    assert gap3 / 90 < gap2 / 8  # 90 units of TF gained


# ── Document-Length Normalization ────────────────────────────

def test_length_normalization():
    """Same TF, shorter doc → higher score (with b > 0)."""
    idf = bm25_idf(100, 10)

    short = bm25_term_score(
        term_frequency=2, document_length=10,
        average_document_length=50.0, idf=idf,
        k1=1.2, b=0.75,
    )
    long = bm25_term_score(
        term_frequency=2, document_length=100,
        average_document_length=50.0, idf=idf,
        k1=1.2, b=0.75,
    )
    assert short > long


def test_length_normalization_b_zero():
    """With b=0, document length has no effect."""
    idf = bm25_idf(100, 10)

    short = bm25_term_score(
        term_frequency=2, document_length=10,
        average_document_length=50.0, idf=idf,
        k1=1.2, b=0,
    )
    long = bm25_term_score(
        term_frequency=2, document_length=100,
        average_document_length=50.0, idf=idf,
        k1=1.2, b=0,
    )
    assert math.isclose(short, long)


# ── k1 Parameter ─────────────────────────────────────────────

def test_k1_zero():
    """k1=0 → frequency has minimal effect beyond presence."""
    idf = bm25_idf(100, 10)

    score1 = bm25_term_score(
        term_frequency=1, document_length=10,
        average_document_length=10.0, idf=idf,
        k1=0, b=0.75,
    )
    score100 = bm25_term_score(
        term_frequency=100, document_length=10,
        average_document_length=10.0, idf=idf,
        k1=0, b=0.75,
    )
    # With k1=0, both should be equal (numerator = f*1, denom = f)
    assert math.isclose(score1, score100)


def test_k1_affects_saturation():
    """Higher k1 → more differentiation between TF values."""
    idf = bm25_idf(100, 10)

    def score_diff(k1: float) -> float:
        s1 = bm25_term_score(
            term_frequency=1, document_length=10,
            average_document_length=10.0, idf=idf,
            k1=k1, b=0.75,
        )
        s10 = bm25_term_score(
            term_frequency=10, document_length=10,
            average_document_length=10.0, idf=idf,
            k1=k1, b=0.75,
        )
        return s10 - s1

    # Higher k1 → larger gap between TF=1 and TF=10
    diff_low = score_diff(0.5)
    diff_high = score_diff(2.0)
    assert diff_high > diff_low


def test_k1_invalid_negative():
    with pytest.raises(ValueError, match="k1 must be >= 0"):
        bm25_term_score(
            term_frequency=1, document_length=10,
            average_document_length=10.0, idf=1.0,
            k1=-1, b=0.75,
        )


# ── b Parameter ──────────────────────────────────────────────

def test_b_values():
    """b=0, 0.5, 0.75, 1.0 all produce valid scores."""
    idf = bm25_idf(100, 10)
    for b_val in [0, 0.5, 0.75, 1.0]:
        score = bm25_term_score(
            term_frequency=2, document_length=10,
            average_document_length=10.0, idf=idf,
            k1=1.2, b=b_val,
        )
        assert score > 0


def test_b_affects_length_sensitivity():
    """Higher b → more sensitivity to document length."""
    idf = bm25_idf(100, 10)

    def length_ratio(b_val: float) -> float:
        short = bm25_term_score(
            term_frequency=2, document_length=5,
            average_document_length=50.0, idf=idf,
            k1=1.2, b=b_val,
        )
        long = bm25_term_score(
            term_frequency=2, document_length=100,
            average_document_length=50.0, idf=idf,
            k1=1.2, b=b_val,
        )
        return short / long

    # Higher b → larger ratio (short doc benefits more)
    assert length_ratio(1.0) > length_ratio(0.5)
    assert length_ratio(0.5) > length_ratio(0.0)


def test_b_invalid_negative():
    with pytest.raises(ValueError, match="b must be in"):
        bm25_term_score(
            term_frequency=1, document_length=10,
            average_document_length=10.0, idf=1.0,
            k1=1.2, b=-0.1,
        )


def test_b_invalid_above_1():
    with pytest.raises(ValueError, match="b must be in"):
        bm25_term_score(
            term_frequency=1, document_length=10,
            average_document_length=10.0, idf=1.0,
            k1=1.2, b=1.5,
        )


# ── Other Validation ─────────────────────────────────────────

def test_invalid_negative_tf():
    with pytest.raises(ValueError, match="term_frequency must be >= 0"):
        bm25_term_score(
            term_frequency=-1, document_length=10,
            average_document_length=10.0, idf=1.0,
        )


def test_invalid_negative_doc_length():
    with pytest.raises(ValueError, match="document_length must be >= 0"):
        bm25_term_score(
            term_frequency=1, document_length=-5,
            average_document_length=10.0, idf=1.0,
        )


def test_invalid_zero_avgdl():
    with pytest.raises(ValueError, match="average_document_length must be > 0"):
        bm25_term_score(
            term_frequency=1, document_length=10,
            average_document_length=0, idf=1.0,
        )


# ══════════════════════════════════════════════════════════════
# BM25 Complete Score
# ══════════════════════════════════════════════════════════════

def test_bm25_score_single_term():
    idf = bm25_idf(100, 10)
    score = bm25_score(
        term_frequencies={"python": 3},
        document_length=10,
        average_document_length=10.0,
        idfs={"python": idf},
    )
    assert score > 0


def test_bm25_score_multi_term():
    idf_p = bm25_idf(100, 10)
    idf_w = bm25_idf(100, 5)
    score = bm25_score(
        term_frequencies={"python": 2, "web": 1},
        document_length=10,
        average_document_length=10.0,
        idfs={"python": idf_p, "web": idf_w},
    )
    # Should be > single-term score
    single = bm25_score(
        term_frequencies={"python": 2},
        document_length=10,
        average_document_length=10.0,
        idfs={"python": idf_p},
    )
    assert score > single


def test_bm25_score_empty():
    score = bm25_score(
        term_frequencies={},
        document_length=10,
        average_document_length=10.0,
        idfs={},
    )
    assert score == 0.0


# ── Hand-calculated BM25 test (mandatory) ────────────────────

def test_hand_calculated_bm25():
    """
    Manual calculation:
    N=4, DF=1, TF=1, |D|=3, avgdl=4, k1=1.2, b=0.75

    IDF = log(1 + (4-1+0.5)/(1+0.5))
        = log(1 + 3.5/1.5)
        = log(1 + 2.3333...)
        = log(3.3333...)

    norm = 1 - 0.75 + 0.75 × 3/4 = 1 - 0.75 + 0.5625 = 0.8125
    denom = 1 + 1.2 × 0.8125 = 1 + 0.975 = 1.975
    numer = 1 × (1.2 + 1) = 2.2
    term_score = IDF × 2.2 / 1.975
    """
    idf = bm25_idf(4, 1)
    expected_idf = math.log(1 + 3.5 / 1.5)
    assert math.isclose(idf, expected_idf)

    score = bm25_term_score(
        term_frequency=1,
        document_length=3,
        average_document_length=4.0,
        idf=idf,
        k1=1.2,
        b=0.75,
    )

    norm = 1.0 - 0.75 + 0.75 * 3.0 / 4.0  # 0.8125
    denom = 1.0 + 1.2 * norm  # 1.975
    numer = 1.0 * 2.2  # 2.2
    expected = expected_idf * numer / denom
    assert math.isclose(score, expected)
