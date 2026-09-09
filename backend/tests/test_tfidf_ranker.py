"""
Tests for the TF-IDF ranker.

Covers:
- One-term query
- Multi-term query
- Rare terms rank higher
- Common terms rank lower
- Document frequency differences
- Document length normalization (cosine)
- Duplicate query terms
- Unknown terms
- Empty candidate set
- Deterministic ordering
- Equal-score doc-ID tie-breaking
- Zero-score documents
- Hand-calculated test
"""

import math

from app.indexing.inverted_index import InvertedIndex
from app.search.tfidf import tf_idf
from app.search.tfidf_ranker import rank_candidates


# ── helpers ──────────────────────────────────────────────────

def _build_corpus() -> InvertedIndex:
    """
    4-document corpus:
    Doc 1: python is a programming language
    Doc 2: javascript is a programming language for the web
    Doc 3: python can be used for web development
    Doc 4: databases store structured information
    """
    idx = InvertedIndex()
    idx.add_document(1, ["python", "is", "a", "programming", "language"])
    idx.add_document(2, ["javascript", "is", "a", "programming", "language",
                         "for", "the", "web"])
    idx.add_document(3, ["python", "can", "be", "used", "for", "web",
                         "development"])
    idx.add_document(4, ["databases", "store", "structured", "information"])
    return idx


# ── One-term query ───────────────────────────────────────────

def test_one_term_query():
    idx = _build_corpus()
    results = rank_candidates(["python"], [1, 3], idx)

    assert len(results) == 2
    doc_ids = [r.document_id for r in results]
    assert 1 in doc_ids
    assert 3 in doc_ids


def test_one_term_scores_positive():
    idx = _build_corpus()
    results = rank_candidates(["python"], [1, 3], idx)

    for r in results:
        assert r.score > 0


# ── Multi-term query ────────────────────────────────────────

def test_multi_term_query():
    idx = _build_corpus()
    results = rank_candidates(
        ["python", "web"], [1, 2, 3], idx
    )

    assert len(results) == 3
    # Doc 3 matches both python AND web — should rank highest
    assert results[0].document_id == 3


# ── Rare terms rank higher ──────────────────────────────────

def test_rare_term_ranks_higher():
    """
    'databases' appears in only doc 4 (very rare).
    'is' appears in docs 1 and 2 (common).
    Searching for both should rank doc 4 first.
    """
    idx = _build_corpus()
    results = rank_candidates(
        ["databases", "is"], [1, 2, 4], idx
    )

    # Doc 4 has the rare term "databases" — highest IDF
    assert results[0].document_id == 4


def test_idf_ordering():
    """Verify rare terms have greater influence."""
    idx = _build_corpus()

    # "databases" only in doc 4 (DF=1)
    # "programming" in docs 1, 2 (DF=2)
    # "is" in docs 1, 2 (DF=2)
    results_rare = rank_candidates(["databases"], [4], idx)
    results_common = rank_candidates(["programming"], [1], idx)

    # Rare term doc should have higher score
    assert results_rare[0].score > results_common[0].score


# ── Document frequency differences ──────────────────────────

def test_df_affects_ranking():
    """Two docs matching same query term — different contexts matter."""
    idx = InvertedIndex()
    idx.add_document(1, ["alpha", "beta", "gamma"])
    idx.add_document(2, ["alpha", "delta", "epsilon"])
    idx.add_document(3, ["alpha", "beta", "zeta"])

    # Query "alpha beta" — docs 1, 3 match both; doc 2 only matches alpha
    results = rank_candidates(
        ["alpha", "beta"], [1, 2, 3], idx
    )

    # Docs 1 and 3 match both terms and should rank above doc 2
    top_two_ids = {results[0].document_id, results[1].document_id}
    assert top_two_ids == {1, 3}


# ── Document length normalization (cosine) ───────────────────

def test_length_normalization():
    """
    Document A: python × 5 (long, all same term)
    Document B: python × 1 (short)

    Query: python

    Cosine similarity should normalize for document length.
    Both documents only contain "python", so the cosine with a
    query vector of just "python" should be 1.0 for both (same
    direction in vector space).
    """
    idx = InvertedIndex()
    idx.add_document(1, ["python"] * 5)
    idx.add_document(2, ["python"])

    results = rank_candidates(["python"], [1, 2], idx)

    # Both are pure "python" vectors — cosine should be the same
    assert math.isclose(results[0].score, results[1].score, rel_tol=1e-9)


def test_length_normalization_mixed():
    """
    Doc A: python + many filler terms shared with other docs
    Doc B: python only

    Query: python

    Doc B is purely about python (cosine ≈ 1.0).
    Doc A has filler terms diluting the python signal, so
    cosine(Q, A) < cosine(Q, B).
    """
    idx = InvertedIndex()
    # "filler" appears in both docs, so it has low IDF and doesn't help.
    # But it still adds magnitude to doc A's vector.
    idx.add_document(1, ["python", "filler", "filler", "filler", "filler"])
    idx.add_document(2, ["python"])
    idx.add_document(3, ["filler"])  # So filler DF > 1

    results = rank_candidates(["python"], [1, 2], idx)

    # Doc 2 (pure python) should rank higher — better cosine alignment
    assert results[0].document_id == 2
    assert results[0].score > results[1].score


# ── Duplicate query terms ────────────────────────────────────

def test_duplicate_query_terms():
    """Duplicates must not inflate scores."""
    idx = _build_corpus()
    results_single = rank_candidates(["python"], [1, 3], idx)
    results_dup = rank_candidates(["python", "python", "python"], [1, 3], idx)

    for r1, r2 in zip(results_single, results_dup):
        assert r1.document_id == r2.document_id
        assert math.isclose(r1.score, r2.score)


# ── Unknown terms ────────────────────────────────────────────

def test_unknown_terms_only():
    idx = _build_corpus()
    results = rank_candidates(["zzznonsense"], [1, 2], idx)
    assert results == []


def test_mix_known_unknown():
    idx = _build_corpus()
    results = rank_candidates(["python", "zzznonsense"], [1, 3], idx)

    # Should still return results for "python"
    assert len(results) == 2
    for r in results:
        assert r.score > 0


# ── Empty candidate set ─────────────────────────────────────

def test_empty_candidates():
    idx = _build_corpus()
    results = rank_candidates(["python"], [], idx)
    assert results == []


def test_empty_query():
    idx = _build_corpus()
    results = rank_candidates([], [1, 2], idx)
    assert results == []


# ── Deterministic ordering ───────────────────────────────────

def test_deterministic():
    idx = _build_corpus()
    first = rank_candidates(["python", "web"], [1, 2, 3], idx)

    for _ in range(50):
        result = rank_candidates(["python", "web"], [1, 2, 3], idx)
        ids = [r.document_id for r in result]
        assert ids == [r.document_id for r in first]


# ── Equal-score doc-ID tie-breaking ──────────────────────────

def test_doc_id_tiebreaker():
    """Documents with equal scores should be ordered by ID ascending."""
    idx = InvertedIndex()
    idx.add_document(5, ["python"])
    idx.add_document(2, ["python"])
    idx.add_document(8, ["python"])

    results = rank_candidates(["python"], [5, 2, 8], idx)

    # All have identical scores — order by doc_id
    ids = [r.document_id for r in results]
    assert ids == [2, 5, 8]


# ── Zero-score documents ────────────────────────────────────

def test_zero_score_candidate():
    """
    If a candidate is supplied but the query terms don't overlap
    with it (shouldn't happen in practice, but must be safe).
    """
    idx = InvertedIndex()
    idx.add_document(1, ["python"])
    idx.add_document(2, ["javascript"])

    # Candidate doc 2 doesn't contain "python"
    results = rank_candidates(["python"], [1, 2], idx)

    # Doc 2 should have score 0 (no overlap with query)
    by_id = {r.document_id: r for r in results}
    assert by_id[2].score == 0.0
    assert by_id[1].score > 0.0


# ── Matched terms tracking ──────────────────────────────────

def test_matched_terms():
    idx = _build_corpus()
    results = rank_candidates(["python", "web"], [1, 2, 3], idx)

    by_id = {r.document_id: r for r in results}

    assert "python" in by_id[1].matched_terms
    assert "web" not in by_id[1].matched_terms  # doc 1 has no "web"

    assert "python" in by_id[3].matched_terms
    assert "web" in by_id[3].matched_terms


# ── Hand-calculated TF-IDF cosine test ───────────────────────

def test_hand_calculated_ranking():
    """
    Tiny corpus for manual verification.

    Doc 1: ["a", "b"]
    Doc 2: ["a"]

    N = 2
    DF("a") = 2, DF("b") = 1

    Query: ["a", "b"]

    IDF("a") = log(2/2) = 0  (appears in all docs)
    IDF("b") = log(2/1) = log(2)

    Query vector: {"a": 0.0, "b": log(2)} → effectively {"b": log(2)}

    Doc 1 vector:
      "a": TF-IDF = (1+log(1)) × log(2/2) = 1 × 0 = 0
      "b": TF-IDF = (1+log(1)) × log(2/1) = 1 × log(2) = log(2)
      → {"b": log(2)}

    Doc 2 vector:
      "a": TF-IDF = (1+log(1)) × log(2/2) = 0
      → {} (effectively zero)

    cos(Q, D1) = (log(2)*log(2)) / (log(2) * log(2)) = 1.0
    cos(Q, D2) = 0.0

    Doc 1 should rank first with score ≈ 1.0
    Doc 2 should have score 0.0
    """
    idx = InvertedIndex()
    idx.add_document(1, ["a", "b"])
    idx.add_document(2, ["a"])

    results = rank_candidates(["a", "b"], [1, 2], idx)

    assert results[0].document_id == 1
    assert math.isclose(results[0].score, 1.0)
    assert results[1].document_id == 2
    assert results[1].score == 0.0
