"""
Tests for the BM25 ranker.

Covers:
- One-term / multi-term query
- Rare terms rank higher
- Document frequency differences
- TF saturation (diminishing returns)
- Document-length normalization
- b=0 disables length normalization
- Duplicate query terms
- Unknown terms
- Empty candidate set / empty query
- Deterministic ordering
- Doc-ID tie-breaking
- Zero-score documents
- Parameter sensitivity (k1, b)
- Hand-calculated test
- BM25 vs TF-IDF ranking comparison
"""

import math

from app.indexing.inverted_index import InvertedIndex
from app.search.bm25_ranker import rank_candidates
from app.search.tfidf_ranker import rank_candidates as tfidf_rank


# ── helpers ──────────────────────────────────────────────────

def _build_corpus() -> InvertedIndex:
    """
    4-document corpus:
    Doc 1: python is a programming language          (5 tokens)
    Doc 2: javascript is a programming language for the web (8 tokens)
    Doc 3: python can be used for web development    (7 tokens)
    Doc 4: databases store structured information    (4 tokens)
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
    assert {r.document_id for r in results} == {1, 3}


def test_one_term_scores_positive():
    idx = _build_corpus()
    results = rank_candidates(["python"], [1, 3], idx)

    for r in results:
        assert r.score > 0


# ── Multi-term query ────────────────────────────────────────

def test_multi_term_query():
    idx = _build_corpus()
    results = rank_candidates(["python", "web"], [1, 2, 3], idx)

    assert len(results) == 3
    # Doc 3 matches both python AND web — should rank highest
    assert results[0].document_id == 3


# ── Rare terms rank higher ──────────────────────────────────

def test_rare_term_ranks_higher():
    """'databases' (DF=1) is much rarer than 'is' (DF=2)."""
    idx = _build_corpus()
    results = rank_candidates(["databases", "is"], [1, 2, 4], idx)

    # Doc 4 has the rare term "databases"
    assert results[0].document_id == 4


# ── TF Saturation (diminishing returns) ─────────────────────

def test_tf_saturation_ranking():
    """
    Doc 1: python (TF=1)
    Doc 2: python python python (TF=3)
    Doc 3: python × 100 (TF=100)

    Scores increase with TF but with diminishing returns.
    """
    idx = InvertedIndex()
    idx.add_document(1, ["python"])
    idx.add_document(2, ["python", "python", "python"])
    idx.add_document(3, ["python"] * 100)

    results = rank_candidates(["python"], [1, 2, 3], idx)

    by_id = {r.document_id: r for r in results}

    # Scores increase
    assert by_id[2].score > by_id[1].score
    assert by_id[3].score > by_id[2].score

    # Diminishing returns: gap per unit shrinks
    gap_1_to_3 = by_id[2].score - by_id[1].score  # 2 extra TF
    gap_3_to_100 = by_id[3].score - by_id[2].score  # 97 extra TF
    assert gap_3_to_100 / 97 < gap_1_to_3 / 2


# ── Document-length normalization ────────────────────────────

def test_length_normalization():
    """Same TF, shorter doc ranks higher (with b=0.75)."""
    idx = InvertedIndex()
    idx.add_document(1, ["python"] + ["filler"] * 50)  # long doc
    idx.add_document(2, ["python"])  # short doc
    idx.add_document(3, ["filler"] * 10)  # so filler has DF > 1

    results = rank_candidates(["python"], [1, 2], idx, b=0.75)

    assert results[0].document_id == 2  # short doc first
    assert results[0].score > results[1].score


def test_length_normalization_b_zero():
    """With b=0, document length has no effect on ranking."""
    idx = InvertedIndex()
    idx.add_document(1, ["python"] + ["other"] * 50)
    idx.add_document(2, ["python"])

    results_b0 = rank_candidates(["python"], [1, 2], idx, b=0)

    # With b=0, TF is the only differentiator. Both have TF=1.
    # So scores should be equal (k1 normalization still applies
    # but identically for both since TF is the same).
    assert math.isclose(results_b0[0].score, results_b0[1].score)


# ── Duplicate query terms ────────────────────────────────────

def test_duplicate_query_terms():
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

    assert len(results) == 2
    for r in results:
        assert r.score > 0


# ── Empty inputs ─────────────────────────────────────────────

def test_empty_candidates():
    idx = _build_corpus()
    results = rank_candidates(["python"], [], idx)
    assert results == []


def test_empty_query():
    idx = _build_corpus()
    results = rank_candidates([], [1, 2], idx)
    assert results == []


def test_empty_index():
    idx = InvertedIndex()
    results = rank_candidates(["python"], [1], idx)
    assert results == []


# ── Deterministic ordering ───────────────────────────────────

def test_deterministic():
    idx = _build_corpus()
    first = rank_candidates(["python", "web"], [1, 2, 3], idx)

    for _ in range(50):
        result = rank_candidates(["python", "web"], [1, 2, 3], idx)
        ids = [r.document_id for r in result]
        assert ids == [r.document_id for r in first]


# ── Doc-ID tie-breaking ─────────────────────────────────────

def test_doc_id_tiebreaker():
    idx = InvertedIndex()
    idx.add_document(5, ["python"])
    idx.add_document(2, ["python"])
    idx.add_document(8, ["python"])

    results = rank_candidates(["python"], [5, 2, 8], idx)
    ids = [r.document_id for r in results]
    assert ids == [2, 5, 8]


# ── Zero-score candidate ────────────────────────────────────

def test_zero_score_candidate():
    idx = InvertedIndex()
    idx.add_document(1, ["python"])
    idx.add_document(2, ["javascript"])

    results = rank_candidates(["python"], [1, 2], idx)

    by_id = {r.document_id: r for r in results}
    assert by_id[2].score == 0.0
    assert by_id[1].score > 0.0


# ── Matched terms tracking ──────────────────────────────────

def test_matched_terms():
    idx = _build_corpus()
    results = rank_candidates(["python", "web"], [1, 2, 3], idx)

    by_id = {r.document_id: r for r in results}
    assert "python" in by_id[1].matched_terms
    assert "web" not in by_id[1].matched_terms

    assert "python" in by_id[3].matched_terms
    assert "web" in by_id[3].matched_terms


# ── Parameter sensitivity ───────────────────────────────────

def test_k1_sensitivity():
    """Different k1 values produce different scores."""
    idx = _build_corpus()
    results_low = rank_candidates(["python"], [1, 3], idx, k1=0.5)
    results_high = rank_candidates(["python"], [1, 3], idx, k1=2.0)

    # Same ordering but different scores
    assert not math.isclose(
        results_low[0].score, results_high[0].score
    )


def test_b_sensitivity():
    """Different b values produce different scores for different-length docs."""
    idx = InvertedIndex()
    idx.add_document(1, ["python"] + ["filler"] * 20)
    idx.add_document(2, ["python"])

    results_b0 = rank_candidates(["python"], [1, 2], idx, b=0)
    results_b1 = rank_candidates(["python"], [1, 2], idx, b=1)

    # b=0 gives equal scores; b=1 gives different scores
    assert math.isclose(results_b0[0].score, results_b0[1].score)
    assert not math.isclose(results_b1[0].score, results_b1[1].score)


# ── BM25 vs TF-IDF produces different rankings ──────────────

def test_bm25_vs_tfidf_different():
    """
    Demonstrate BM25 and TF-IDF produce different scores.

    Use a corpus where document length matters:
    Doc 1: python × 10 (short, repetitive)
    Doc 2: python + filler × 50 (long, diluted)
    """
    idx = InvertedIndex()
    idx.add_document(1, ["python"] * 10)
    idx.add_document(2, ["python"] + ["filler"] * 50)
    idx.add_document(3, ["filler"] * 10)

    bm25_results = rank_candidates(["python"], [1, 2], idx)
    tfidf_results = tfidf_rank(["python"], [1, 2], idx)

    # Both algorithms rank the same candidates, but scores differ
    bm25_scores = {r.document_id: r.score for r in bm25_results}
    tfidf_scores = {r.document_id: r.score for r in tfidf_results}

    assert not math.isclose(bm25_scores[1], tfidf_scores[1])


# ── Hand-calculated BM25 ranking test ────────────────────────

def test_hand_calculated_ranking():
    """
    Tiny corpus — verify actual BM25 scores.

    Doc 1: ["a", "b"]           len=2
    Doc 2: ["a"]                len=1
    N=2, avgdl=1.5

    Query: ["a"]
    DF("a") = 2

    IDF("a") = log(1 + (2-2+0.5)/(2+0.5)) = log(1 + 0.5/2.5) = log(1.2)

    Doc 1 (len=2):
      norm = 1 - 0.75 + 0.75 * 2/1.5 = 0.25 + 1.0 = 1.25
      denom = 1 + 1.2 * 1.25 = 2.5
      numer = 1 * 2.2 = 2.2
      score = log(1.2) × 2.2 / 2.5

    Doc 2 (len=1):
      norm = 1 - 0.75 + 0.75 * 1/1.5 = 0.25 + 0.5 = 0.75
      denom = 1 + 1.2 * 0.75 = 1.9
      numer = 1 * 2.2 = 2.2
      score = log(1.2) × 2.2 / 1.9
    """
    idx = InvertedIndex()
    idx.add_document(1, ["a", "b"])
    idx.add_document(2, ["a"])

    results = rank_candidates(["a"], [1, 2], idx, k1=1.2, b=0.75)

    idf_a = math.log(1.2)

    expected_1 = idf_a * 2.2 / 2.5
    expected_2 = idf_a * 2.2 / 1.9

    by_id = {r.document_id: r for r in results}
    assert math.isclose(by_id[1].score, expected_1, rel_tol=1e-9)
    assert math.isclose(by_id[2].score, expected_2, rel_tol=1e-9)

    # Shorter doc should rank higher
    assert results[0].document_id == 2
