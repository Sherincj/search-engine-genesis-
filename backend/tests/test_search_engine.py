"""
Tests for the SearchEngine (ranking logic).

Covers:
- One-term query
- Multi-term query
- Matched-term priority (more unique matches ranks higher)
- TF tie-breaking
- Document-ID tie-breaking
- Unknown terms
- Duplicate query terms
- Empty query
- Deterministic ordering
"""

from app.indexing.inverted_index import InvertedIndex
from app.search.search_engine import SearchEngine


def _build_ranking_index() -> InvertedIndex:
    """
    Build an index for ranking tests:

    Doc 1: python python           → python TF=2
    Doc 2: python javascript       → python TF=1, javascript TF=1
    Doc 3: javascript javascript javascript → javascript TF=3
    """
    idx = InvertedIndex()
    idx.add_document(1, ["python", "python"])
    idx.add_document(2, ["python", "javascript"])
    idx.add_document(3, ["javascript", "javascript", "javascript"])
    return idx


# ── One-term query ───────────────────────────────────────────

def test_one_term_query():
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("python")

    doc_ids = [c.document_id for c in result.candidates]
    assert 1 in doc_ids
    assert 2 in doc_ids


def test_one_term_ranking_by_tf():
    """For a single query term, higher TF ranks higher."""
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("python")

    doc_ids = [c.document_id for c in result.candidates]
    # Doc 1 has TF=2, Doc 2 has TF=1
    assert doc_ids.index(1) < doc_ids.index(2)


# ── Multi-term query ────────────────────────────────────────

def test_multi_term_ranking():
    """
    Query: python javascript

    Doc 2: matches both terms → rank 1 (matched_terms=2)
    Doc 3: matches javascript only, TF=3 → rank 2
    Doc 1: matches python only, TF=2 → rank 3
    """
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("python javascript")

    doc_ids = [c.document_id for c in result.candidates]
    assert doc_ids == [2, 3, 1]


# ── Matched-term priority ───────────────────────────────────

def test_matched_term_priority_over_tf():
    """A document matching 2 terms always outranks one matching 1 term,
    even if the single-term match has higher TF."""
    idx = InvertedIndex()
    idx.add_document(1, ["python"] * 100)  # TF=100, but only 1 matched term
    idx.add_document(2, ["python", "web"])  # TF=2 total, but 2 matched terms

    engine = SearchEngine(idx)
    result = engine.search("python web")

    doc_ids = [c.document_id for c in result.candidates]
    assert doc_ids[0] == 2  # 2 matched terms beats TF=100


# ── TF tie-breaking ─────────────────────────────────────────

def test_tf_tie_breaking():
    """Same matched_term_count → higher TF wins."""
    idx = InvertedIndex()
    idx.add_document(1, ["python"])
    idx.add_document(2, ["python", "python", "python"])

    engine = SearchEngine(idx)
    result = engine.search("python")

    doc_ids = [c.document_id for c in result.candidates]
    assert doc_ids == [2, 1]


# ── Document-ID tie-breaking ────────────────────────────────

def test_doc_id_tie_breaking():
    """Same matched_term_count and same TF → lower doc_id first."""
    idx = InvertedIndex()
    idx.add_document(5, ["python"])
    idx.add_document(2, ["python"])
    idx.add_document(8, ["python"])

    engine = SearchEngine(idx)
    result = engine.search("python")

    doc_ids = [c.document_id for c in result.candidates]
    assert doc_ids == [2, 5, 8]


# ── Unknown terms ────────────────────────────────────────────

def test_unknown_term():
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("rust")

    assert result.total == 0
    assert result.candidates == []


def test_mix_known_unknown():
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("python rust")

    # Only python-matching docs should appear
    doc_ids = {c.document_id for c in result.candidates}
    assert doc_ids == {1, 2}


# ── Duplicate query terms ────────────────────────────────────

def test_duplicate_query_terms():
    """Duplicates must not inflate matched_term_count."""
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("python python python")

    for c in result.candidates:
        assert c.matched_term_count == 1


# ── Empty query ──────────────────────────────────────────────

def test_empty_query():
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("")

    assert result.total == 0
    assert result.candidates == []


def test_whitespace_query():
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("   ")

    assert result.total == 0
    assert result.candidates == []


# ── Deterministic ordering ───────────────────────────────────

def test_deterministic():
    engine = SearchEngine(_build_ranking_index())

    results = [engine.search("python javascript") for _ in range(50)]
    first = [c.document_id for c in results[0].candidates]

    for r in results:
        assert [c.document_id for c in r.candidates] == first


# ── Pagination ───────────────────────────────────────────────

def test_pagination_limit():
    idx = InvertedIndex()
    for i in range(1, 11):
        idx.add_document(i, ["python"])

    engine = SearchEngine(idx)
    result = engine.search("python", limit=3, offset=0)

    assert result.total == 10
    assert len(result.candidates) == 3


def test_pagination_offset():
    idx = InvertedIndex()
    for i in range(1, 11):
        idx.add_document(i, ["python"])

    engine = SearchEngine(idx)
    page1 = engine.search("python", limit=3, offset=0)
    page2 = engine.search("python", limit=3, offset=3)

    ids1 = [c.document_id for c in page1.candidates]
    ids2 = [c.document_id for c in page2.candidates]

    # No overlap
    assert not set(ids1) & set(ids2)
    # Different pages
    assert len(ids2) == 3


def test_pagination_last_page():
    idx = InvertedIndex()
    for i in range(1, 6):
        idx.add_document(i, ["python"])

    engine = SearchEngine(idx)
    result = engine.search("python", limit=3, offset=3)

    assert result.total == 5
    assert len(result.candidates) == 2  # Only 2 left


# ── Result metadata ──────────────────────────────────────────

def test_result_metadata():
    engine = SearchEngine(_build_ranking_index())
    result = engine.search("python")

    assert result.query == "python"
    assert result.took_ms >= 0
    assert isinstance(result.total, int)


# ── Empty index ──────────────────────────────────────────────

def test_empty_index():
    engine = SearchEngine(InvertedIndex())
    result = engine.search("python")

    assert result.total == 0
    assert result.candidates == []
