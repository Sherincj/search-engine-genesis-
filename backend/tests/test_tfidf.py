"""
Tests for TF-IDF mathematical functions.

Covers:
- Term frequency (logarithmic)
- Inverse document frequency
- TF-IDF combined weight
- Cosine similarity
- Input validation
- Hand-calculated values
- Document term accessor
"""

import math

import pytest

from app.indexing.inverted_index import InvertedIndex
from app.search.tfidf import (
    cosine_similarity,
    inverse_document_frequency,
    term_frequency,
    tf_idf,
)


# ══════════════════════════════════════════════════════════════
# Term Frequency
# ══════════════════════════════════════════════════════════════

def test_tf_frequency_1():
    """TF(1) = 1 + log(1) = 1."""
    assert term_frequency(1) == 1.0


def test_tf_frequency_growth():
    """TF must increase with frequency."""
    assert term_frequency(2) > term_frequency(1)
    assert term_frequency(10) > term_frequency(2)
    assert term_frequency(100) > term_frequency(10)


def test_tf_logarithmic_not_linear():
    """Growth must be sub-linear (logarithmic)."""
    tf1 = term_frequency(1)
    tf5 = term_frequency(5)
    tf25 = term_frequency(25)

    # If linear: tf25 - tf5 would equal tf5 - tf1 × (20/4) = 5x
    # Logarithmic: the absolute gap shrinks relative to the interval.
    gap_low = tf5 - tf1       # log(5) - log(1) = log(5)
    gap_high = tf25 - tf5     # log(25) - log(5) = log(5)
    # Same multiplicative ratio → same log gap.
    # Use additive interval check instead:
    # tf(2) - tf(1) vs tf(11) - tf(10)
    gap_a = term_frequency(2) - term_frequency(1)   # log(2)
    gap_b = term_frequency(11) - term_frequency(10)  # log(11/10) ≈ 0.095
    assert gap_b < gap_a


def test_tf_exact_value_2():
    """TF(2) = 1 + log(2)."""
    assert math.isclose(term_frequency(2), 1.0 + math.log(2))


def test_tf_exact_value_10():
    """TF(10) = 1 + log(10)."""
    assert math.isclose(term_frequency(10), 1.0 + math.log(10))


def test_tf_invalid_zero():
    with pytest.raises(ValueError, match="must be >= 1"):
        term_frequency(0)


def test_tf_invalid_negative():
    with pytest.raises(ValueError, match="must be >= 1"):
        term_frequency(-5)


# ══════════════════════════════════════════════════════════════
# Inverse Document Frequency
# ══════════════════════════════════════════════════════════════

def test_idf_rare_term():
    """N=100, DF=1 → high IDF."""
    idf = inverse_document_frequency(100, 1)
    assert idf == math.log(100)


def test_idf_common_term():
    """N=100, DF=50 → lower IDF."""
    idf = inverse_document_frequency(100, 50)
    assert idf == math.log(2)


def test_idf_all_documents():
    """N=100, DF=100 → IDF = 0."""
    idf = inverse_document_frequency(100, 100)
    assert idf == 0.0


def test_idf_rare_greater_than_common():
    """Rare terms must have higher IDF than common terms."""
    idf_rare = inverse_document_frequency(100, 1)
    idf_common = inverse_document_frequency(100, 50)
    assert idf_rare > idf_common


def test_idf_invalid_n_zero():
    with pytest.raises(ValueError, match="total_documents must be >= 1"):
        inverse_document_frequency(0, 1)


def test_idf_invalid_df_zero():
    with pytest.raises(ValueError, match="document_frequency must be >= 1"):
        inverse_document_frequency(100, 0)


def test_idf_invalid_df_exceeds_n():
    with pytest.raises(ValueError, match="cannot exceed"):
        inverse_document_frequency(10, 20)


def test_idf_invalid_negative_n():
    with pytest.raises(ValueError):
        inverse_document_frequency(-5, 1)


def test_idf_invalid_negative_df():
    with pytest.raises(ValueError):
        inverse_document_frequency(100, -3)


# ══════════════════════════════════════════════════════════════
# TF-IDF
# ══════════════════════════════════════════════════════════════

def test_tfidf_is_tf_times_idf():
    """TF-IDF must equal TF × IDF."""
    freq, n, df = 3, 100, 10
    expected = term_frequency(freq) * inverse_document_frequency(n, df)
    assert math.isclose(tf_idf(freq, n, df), expected)


def test_tfidf_rare_term_higher():
    """A rare term should have higher TF-IDF (same TF)."""
    rare = tf_idf(frequency=1, total_documents=100, document_frequency=1)
    common = tf_idf(frequency=1, total_documents=100, document_frequency=50)
    assert rare > common


def test_tfidf_higher_tf_increases_weight():
    """Higher TF increases TF-IDF when IDF is constant."""
    low = tf_idf(frequency=1, total_documents=100, document_frequency=10)
    high = tf_idf(frequency=5, total_documents=100, document_frequency=10)
    assert high > low


def test_tfidf_all_documents_is_zero():
    """If term appears in every document, TF-IDF = 0."""
    result = tf_idf(frequency=5, total_documents=100, document_frequency=100)
    assert result == 0.0


# ── Hand-calculated test (mandatory) ─────────────────────────

def test_tfidf_hand_calculated():
    """
    Manual calculation:
    N=4, DF=1, freq=1

    TF  = 1 + log(1) = 1.0
    IDF = log(4/1)   = log(4) ≈ 1.3863
    TF-IDF = 1.0 × log(4) ≈ 1.3863
    """
    result = tf_idf(frequency=1, total_documents=4, document_frequency=1)
    expected = 1.0 * math.log(4)
    assert math.isclose(result, expected)


def test_tfidf_hand_calculated_2():
    """
    N=4, DF=2, freq=3

    TF  = 1 + log(3) ≈ 2.0986
    IDF = log(4/2) = log(2) ≈ 0.6931
    TF-IDF ≈ 2.0986 × 0.6931 ≈ 1.4541
    """
    result = tf_idf(frequency=3, total_documents=4, document_frequency=2)
    expected = (1.0 + math.log(3)) * math.log(2)
    assert math.isclose(result, expected)


# ══════════════════════════════════════════════════════════════
# Cosine Similarity
# ══════════════════════════════════════════════════════════════

def test_cosine_identical_vectors():
    """Same vector → 1.0."""
    v = {"python": 1.0}
    assert math.isclose(cosine_similarity(v, v), 1.0)


def test_cosine_orthogonal_vectors():
    """Completely different terms → 0.0."""
    q = {"python": 1.0}
    d = {"javascript": 1.0}
    assert cosine_similarity(q, d) == 0.0


def test_cosine_partial_overlap():
    """Some shared terms → between 0 and 1."""
    q = {"python": 1.0, "web": 1.0}
    d = {"python": 1.0, "javascript": 1.0}
    sim = cosine_similarity(q, d)
    assert 0.0 < sim < 1.0


def test_cosine_symmetry():
    """cos(Q, D) == cos(D, Q)."""
    q = {"python": 2.0, "web": 1.0}
    d = {"python": 1.0, "api": 3.0}
    assert math.isclose(
        cosine_similarity(q, d),
        cosine_similarity(d, q),
    )


def test_cosine_empty_query():
    """Empty query → 0.0."""
    assert cosine_similarity({}, {"python": 1.0}) == 0.0


def test_cosine_empty_document():
    """Empty document → 0.0."""
    assert cosine_similarity({"python": 1.0}, {}) == 0.0


def test_cosine_both_empty():
    """Both empty → 0.0."""
    assert cosine_similarity({}, {}) == 0.0


def test_cosine_hand_calculated():
    """
    Q = {a: 1, b: 1}
    D = {a: 1, c: 1}

    dot = 1*1 = 1
    |Q| = sqrt(1+1) = sqrt(2)
    |D| = sqrt(1+1) = sqrt(2)
    cos = 1 / (sqrt(2)*sqrt(2)) = 1/2 = 0.5
    """
    q = {"a": 1.0, "b": 1.0}
    d = {"a": 1.0, "c": 1.0}
    assert math.isclose(cosine_similarity(q, d), 0.5)


def test_cosine_magnitude_invariant():
    """Cosine similarity is independent of vector magnitude."""
    q = {"python": 1.0}
    d1 = {"python": 1.0}
    d2 = {"python": 100.0}

    # Both should give cosine = 1.0 (same direction)
    assert math.isclose(cosine_similarity(q, d1), 1.0)
    assert math.isclose(cosine_similarity(q, d2), 1.0)


# ══════════════════════════════════════════════════════════════
# InvertedIndex: get_document_terms
# ══════════════════════════════════════════════════════════════

def test_get_document_terms_exists():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web", "python"])

    terms = idx.get_document_terms(1)
    assert terms == frozenset({"python", "web"})


def test_get_document_terms_unknown():
    idx = InvertedIndex()
    terms = idx.get_document_terms(999)
    assert terms == frozenset()


def test_get_document_terms_immutable():
    """Returned frozenset must not allow mutation of internal state."""
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])

    terms = idx.get_document_terms(1)
    assert isinstance(terms, frozenset)

    # Verify internal state is unchanged
    assert idx.get_document_terms(1) == frozenset({"python", "web"})


def test_get_document_terms_multiple_docs():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["javascript", "api"])

    assert idx.get_document_terms(1) == frozenset({"python", "web"})
    assert idx.get_document_terms(2) == frozenset({"javascript", "api"})
