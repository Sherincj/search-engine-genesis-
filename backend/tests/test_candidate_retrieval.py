"""
Tests for candidate retrieval.

Covers:
- Single term
- Multiple terms (OR semantics)
- Unknown term
- Duplicate query terms (no inflation)
- Empty query
- Multiple posting lists
- Match metadata (matched_terms, total_term_frequency)
"""

from app.indexing.field_index import FieldIndex
from app.search.candidate_retrieval import retrieve_candidates


def _build_test_index() -> FieldIndex:
    """
    Build a small test index:
    Doc 1: python web python       (python TF=2, web TF=1)
    Doc 2: python javascript       (python TF=1, javascript TF=1)
    Doc 3: web javascript          (web TF=1, javascript TF=1)
    """
    idx = FieldIndex()
    idx.add_document(1, body_tokens=["python", "web", "python"])
    idx.add_document(2, body_tokens=["python", "javascript"])
    idx.add_document(3, body_tokens=["web", "javascript"])
    return idx


# ── Single term ──────────────────────────────────────────────

def test_single_term():
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, ["python"])

    doc_ids = {c.document_id for c in candidates}
    assert doc_ids == {1, 2}


def test_single_term_match_metadata():
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, ["python"])

    by_id = {c.document_id: c for c in candidates}

    # Doc 1: python appears twice
    assert by_id[1].matched_terms == {"python"}
    assert by_id[1].total_term_frequency == 2
    assert by_id[1].matched_term_count == 1

    # Doc 2: python appears once
    assert by_id[2].matched_terms == {"python"}
    assert by_id[2].total_term_frequency == 1


# ── Multiple terms — OR semantics ────────────────────────────

def test_or_semantics():
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, ["python", "javascript"])

    doc_ids = {c.document_id for c in candidates}
    # All 3 docs contain at least one of the terms
    assert doc_ids == {1, 2, 3}


def test_or_semantics_match_metadata():
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, ["python", "javascript"])

    by_id = {c.document_id: c for c in candidates}

    # Doc 1: only python matched
    assert by_id[1].matched_terms == {"python"}

    # Doc 2: both python and javascript matched
    assert by_id[2].matched_terms == {"python", "javascript"}
    assert by_id[2].matched_term_count == 2

    # Doc 3: only javascript matched
    assert by_id[3].matched_terms == {"javascript"}


# ── Unknown term ─────────────────────────────────────────────

def test_unknown_term():
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, ["rust"])
    assert candidates == []


def test_one_known_one_unknown():
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, ["python", "rust"])

    doc_ids = {c.document_id for c in candidates}
    assert doc_ids == {1, 2}


# ── Duplicate query terms ────────────────────────────────────

def test_duplicate_terms_not_inflated():
    """Duplicate query terms must not inflate matched_term_count."""
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, ["python", "python", "python"])

    by_id = {c.document_id: c for c in candidates}

    # Still only 1 unique matched term
    assert by_id[1].matched_term_count == 1
    assert by_id[1].matched_terms == {"python"}


# ── Empty query ──────────────────────────────────────────────

def test_empty_query():
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, [])
    assert candidates == []


# ── Multiple posting lists ───────────────────────────────────

def test_multiple_posting_lists():
    """web appears in docs 1 and 3; javascript in docs 2 and 3."""
    idx = _build_test_index()
    candidates = retrieve_candidates(idx, ["web", "javascript"])

    by_id = {c.document_id: c for c in candidates}

    assert 1 in by_id  # web only
    assert 2 in by_id  # javascript only
    assert 3 in by_id  # both

    assert by_id[3].matched_term_count == 2


# ── Empty index ──────────────────────────────────────────────

def test_empty_index():
    idx = FieldIndex()
    candidates = retrieve_candidates(idx, ["python"])
    assert len(candidates) == 0
