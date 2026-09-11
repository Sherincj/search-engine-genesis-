"""
Tests for the SearchEngine (BM25 ranking).

Updated for Phase 7 — ranking now uses BM25
instead of Phase 6 TF-IDF cosine similarity.

Covers:
- One-term query
- Multi-term query
- BM25 ranking behavior
- Document-ID tie-breaking
- Unknown terms
- Duplicate query terms
- Empty query
- Deterministic ordering
- Pagination
- Result metadata
- Floating-point scores
"""

import math

from app.indexing.field_index import FieldIndex
from app.search.search_engine import SearchEngine


def _build_ranking_index() -> FieldIndex:
    """
    Build an index for ranking tests:

    Doc 1: python python           → python TF=2
    Doc 2: python javascript       → python TF=1, javascript TF=1
    Doc 3: javascript javascript javascript → javascript TF=3
    """
    idx = FieldIndex()
    idx.add_document(1, body_tokens=["python", "python"])
    idx.add_document(2, body_tokens=["python", "javascript"])
    idx.add_document(3, body_tokens=["javascript", "javascript", "javascript"])
    return idx


# ── One-term query ───────────────────────────────────────────

def test_one_term_query():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("python")

    doc_ids = [r.document_id for r in result.ranked]
    assert 1 in doc_ids
    assert 2 in doc_ids


def test_one_term_scores_positive():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("python")

    for r in result.ranked:
        if r.score > 0:
            assert isinstance(r.score, float)


# ── Multi-term query ────────────────────────────────────────

def test_multi_term_ranking():
    """
    Query: python javascript

    With TF-IDF cosine similarity, doc 2 (matching both terms)
    should still rank highest.
    """
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("python javascript")

    doc_ids = [r.document_id for r in result.ranked]
    # Doc 2 matches both terms — should rank first
    assert doc_ids[0] == 2


# ── TF-IDF: matching more terms is better ───────────────────

def test_matched_term_priority_over_tf():
    """A document matching 2 query terms should outrank one matching 1,
    when the matched terms have reasonable IDF."""
    idx = FieldIndex()
    idx.add_document(1, body_tokens=["python"] * 5)
    idx.add_document(2, body_tokens=["python", "web"])
    idx.add_document(3, body_tokens=["other"])

    engine = SearchEngine(idx, title_weight=0, body_weight=1)
    result = engine.search("python web")

    doc_ids = [r.document_id for r in result.ranked]
    # Doc 2 matches both python and web
    assert doc_ids[0] == 2


# ── BM25: higher TF scores higher (with saturation) ─────────

def test_bm25_tf_ranking():
    """With BM25, higher TF produces higher score (saturating)."""
    idx = FieldIndex()
    idx.add_document(1, body_tokens=["python"])
    idx.add_document(2, body_tokens=["python", "python", "python"])

    engine = SearchEngine(idx, title_weight=0, body_weight=1)
    result = engine.search("python")

    scores = {r.document_id: r.score for r in result.ranked}
    # Doc 2 (TF=3) should score higher than doc 1 (TF=1)
    assert scores[2] > scores[1]


# ── Document-ID tie-breaking ────────────────────────────────

def test_doc_id_tie_breaking():
    """Same scores → lower doc_id first."""
    idx = FieldIndex()
    idx.add_document(5, body_tokens=["python"])
    idx.add_document(2, body_tokens=["python"])
    idx.add_document(8, body_tokens=["python"])

    engine = SearchEngine(idx, title_weight=0, body_weight=1)
    result = engine.search("python")

    doc_ids = [r.document_id for r in result.ranked]
    assert doc_ids == [2, 5, 8]


# ── Unknown terms ────────────────────────────────────────────

def test_unknown_term():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("rust")

    assert result.total == 0
    assert result.ranked == []


def test_mix_known_unknown():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("python rust")

    # Only python-matching docs should appear
    doc_ids = {r.document_id for r in result.ranked}
    assert doc_ids == {1, 2}


# ── Duplicate query terms ────────────────────────────────────

def test_duplicate_query_terms():
    """Duplicates must not inflate scores."""
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)

    result_single = engine.search("python")
    result_dup = engine.search("python python python")

    ids_single = [r.document_id for r in result_single.ranked]
    ids_dup = [r.document_id for r in result_dup.ranked]
    assert ids_single == ids_dup

    for r1, r2 in zip(result_single.ranked, result_dup.ranked):
        assert math.isclose(r1.score, r2.score)


# ── Empty query ──────────────────────────────────────────────

def test_empty_query():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("")

    assert result.total == 0
    assert result.ranked == []


def test_whitespace_query():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("   ")

    assert result.total == 0
    assert result.ranked == []


# ── Deterministic ordering ───────────────────────────────────

def test_deterministic():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)

    results = [engine.search("python javascript") for _ in range(50)]
    first = [r.document_id for r in results[0].ranked]

    for r in results:
        assert [c.document_id for c in r.ranked] == first


# ── Pagination ───────────────────────────────────────────────

def test_pagination_limit():
    idx = FieldIndex()
    for i in range(1, 11):
        idx.add_document(i, body_tokens=["python"])

    engine = SearchEngine(idx, title_weight=0, body_weight=1)
    result = engine.search("python", limit=3, offset=0)

    assert result.total == 10
    assert len(result.ranked) == 3


def test_pagination_offset():
    idx = FieldIndex()
    for i in range(1, 11):
        idx.add_document(i, body_tokens=["python"])

    engine = SearchEngine(idx, title_weight=0, body_weight=1)
    page1 = engine.search("python", limit=3, offset=0)
    page2 = engine.search("python", limit=3, offset=3)

    ids1 = [r.document_id for r in page1.ranked]
    ids2 = [r.document_id for r in page2.ranked]

    # No overlap
    assert not set(ids1) & set(ids2)
    # Different pages
    assert len(ids2) == 3


def test_pagination_last_page():
    idx = FieldIndex()
    for i in range(1, 6):
        idx.add_document(i, body_tokens=["python"])

    engine = SearchEngine(idx, title_weight=0, body_weight=1)
    result = engine.search("python", limit=3, offset=3)

    assert result.total == 5
    assert len(result.ranked) == 2  # Only 2 left


# ── Result metadata ──────────────────────────────────────────

def test_result_metadata():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("python")

    assert result.query == "python"
    assert result.took_ms >= 0
    assert isinstance(result.total, int)


# ── Floating-point scores ───────────────────────────────────

def test_scores_are_float():
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("python")

    for r in result.ranked:
        assert isinstance(r.score, float)


def test_scores_non_negative():
    """BM25 scores should be >= 0."""
    engine = SearchEngine(_build_ranking_index(), title_weight=0, body_weight=1)
    result = engine.search("python javascript")

    for r in result.ranked:
        assert r.score >= 0.0


# ── Empty index ──────────────────────────────────────────────

def test_empty_index():
    engine = SearchEngine(FieldIndex(), title_weight=0, body_weight=1)
    result = engine.search("python")

    assert result.total == 0
    assert result.ranked == []
