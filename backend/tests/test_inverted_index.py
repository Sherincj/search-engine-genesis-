"""
Tests for the core InvertedIndex data structure.

Covers:
- Empty index
- Add single document (TF, positions)
- Add multiple documents (document frequency)
- Document length / average document length
- Duplicate terms (TF accumulation)
- Re-add same document (replaces postings)
- Remove document (postings + stats updated)
- Remove last document for a term (term cleaned up)
- Remove nonexistent document
- Unknown term lookup
- Posting ordering (by doc_id)
- Position ordering (ascending)
- contains_term
- has_document
- clear
- Vocabulary
- Validation — consistent
- Validation — detects inconsistency
- get_posting (direct O(1) lookup)
"""

from app.indexing.inverted_index import InvertedIndex, Posting


# ── Empty index ──────────────────────────────────────────────

def test_empty_index_total_documents():
    idx = InvertedIndex()
    assert idx.get_total_documents() == 0


def test_empty_index_average_length():
    idx = InvertedIndex()
    assert idx.get_average_document_length() == 0.0


def test_empty_index_get_postings():
    idx = InvertedIndex()
    assert idx.get_postings("python") == []


def test_empty_index_vocabulary():
    idx = InvertedIndex()
    assert idx.vocabulary == set()
    assert idx.vocabulary_size == 0


# ── Add single document ─────────────────────────────────────

def test_add_document_postings():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web", "python"])

    postings = idx.get_postings("python")
    assert len(postings) == 1

    p = postings[0]
    assert p.document_id == 1
    assert p.term_frequency == 2
    assert p.positions == [0, 2]


def test_add_document_single_term():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web", "python"])

    postings = idx.get_postings("web")
    assert len(postings) == 1

    p = postings[0]
    assert p.document_id == 1
    assert p.term_frequency == 1
    assert p.positions == [1]


def test_add_document_total_documents():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    assert idx.get_total_documents() == 1


def test_add_document_length():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web", "python"])
    assert idx.get_document_length(1) == 3


# ── Multiple documents — document frequency ─────────────────

def test_document_frequency():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["python", "javascript"])

    assert idx.get_document_frequency("python") == 2
    assert idx.get_document_frequency("web") == 1
    assert idx.get_document_frequency("javascript") == 1


def test_multiple_documents_total():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["python", "javascript"])
    idx.add_document(3, ["web", "javascript"])

    assert idx.get_total_documents() == 3


# ── Document length / average document length ────────────────

def test_document_lengths():
    idx = InvertedIndex()
    idx.add_document(1, ["a", "b", "c"])
    idx.add_document(2, ["x", "y"])

    assert idx.get_document_length(1) == 3
    assert idx.get_document_length(2) == 2


def test_average_document_length():
    idx = InvertedIndex()
    idx.add_document(1, ["a", "b", "c"])
    idx.add_document(2, ["x", "y"])

    assert idx.get_average_document_length() == 2.5


def test_unknown_document_length():
    idx = InvertedIndex()
    assert idx.get_document_length(999) == 0


# ── Duplicate terms (TF accumulation) ────────────────────────

def test_duplicate_terms():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "python", "python"])

    postings = idx.get_postings("python")
    assert len(postings) == 1
    assert postings[0].term_frequency == 3
    assert postings[0].positions == [0, 1, 2]
    assert idx.get_document_frequency("python") == 1


# ── Re-add same document (replace, no duplicates) ────────────

def test_readd_document_replaces():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])

    # Re-add with different tokens
    idx.add_document(1, ["python", "python"])

    # python should have TF=2 now
    postings = idx.get_postings("python")
    assert len(postings) == 1
    assert postings[0].term_frequency == 2
    assert postings[0].positions == [0, 1]


def test_readd_document_removes_old_terms():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])

    # Re-add without "web"
    idx.add_document(1, ["python", "api"])

    # "web" should no longer have doc 1
    assert idx.get_postings("web") == []
    assert not idx.contains_term("web")


def test_readd_preserves_other_documents():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["web", "javascript"])

    # Re-add doc 1
    idx.add_document(1, ["python", "api"])

    # Doc 2's postings should be unchanged
    web_postings = idx.get_postings("web")
    assert len(web_postings) == 1
    assert web_postings[0].document_id == 2

    assert idx.get_total_documents() == 2


# ── Remove document ─────────────────────────────────────────

def test_remove_document():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["python", "javascript"])

    removed = idx.remove_document(1)
    assert removed is True
    assert idx.get_total_documents() == 1


def test_remove_updates_postings():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["python", "javascript"])

    idx.remove_document(1)

    python_postings = idx.get_postings("python")
    assert len(python_postings) == 1
    assert python_postings[0].document_id == 2


def test_remove_updates_document_frequency():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["python"])

    idx.remove_document(1)
    assert idx.get_document_frequency("python") == 1
    assert idx.get_document_frequency("web") == 0


def test_remove_updates_average_length():
    idx = InvertedIndex()
    idx.add_document(1, ["a", "b", "c"])
    idx.add_document(2, ["x", "y"])

    idx.remove_document(1)
    assert idx.get_average_document_length() == 2.0


def test_remove_clears_document_length():
    idx = InvertedIndex()
    idx.add_document(1, ["a", "b"])
    idx.remove_document(1)
    assert idx.get_document_length(1) == 0


# ── Remove last document for a term (cleanup) ───────────────

def test_remove_last_document_cleans_term():
    idx = InvertedIndex()
    idx.add_document(1, ["python"])

    idx.remove_document(1)

    assert not idx.contains_term("python")
    assert idx.get_postings("python") == []
    assert "python" not in idx.vocabulary


# ── Remove nonexistent document ──────────────────────────────

def test_remove_nonexistent_returns_false():
    idx = InvertedIndex()
    assert idx.remove_document(999) is False


# ── Unknown term lookup ──────────────────────────────────────

def test_unknown_term_returns_empty():
    idx = InvertedIndex()
    idx.add_document(1, ["python"])
    assert idx.get_postings("doesnotexist") == []


def test_unknown_term_document_frequency():
    idx = InvertedIndex()
    assert idx.get_document_frequency("doesnotexist") == 0


# ── Posting ordering (by document ID) ────────────────────────

def test_posting_ordering():
    idx = InvertedIndex()
    idx.add_document(5, ["python"])
    idx.add_document(2, ["python"])
    idx.add_document(8, ["python"])
    idx.add_document(1, ["python"])

    postings = idx.get_postings("python")
    doc_ids = [p.document_id for p in postings]
    assert doc_ids == [1, 2, 5, 8]


# ── Position ordering (ascending) ────────────────────────────

def test_position_ordering():
    idx = InvertedIndex()
    tokens = ["web", "python", "api", "python", "python"]
    idx.add_document(1, tokens)

    postings = idx.get_postings("python")
    assert postings[0].positions == [1, 3, 4]


# ── contains_term ────────────────────────────────────────────

def test_contains_term_true():
    idx = InvertedIndex()
    idx.add_document(1, ["python"])
    assert idx.contains_term("python") is True


def test_contains_term_false():
    idx = InvertedIndex()
    assert idx.contains_term("python") is False


# ── has_document ─────────────────────────────────────────────

def test_has_document_true():
    idx = InvertedIndex()
    idx.add_document(1, ["python"])
    assert idx.has_document(1) is True


def test_has_document_false():
    idx = InvertedIndex()
    assert idx.has_document(999) is False


# ── clear ────────────────────────────────────────────────────

def test_clear():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["javascript"])

    idx.clear()

    assert idx.get_total_documents() == 0
    assert idx.vocabulary_size == 0
    assert idx.get_average_document_length() == 0.0
    assert idx.get_postings("python") == []


# ── Vocabulary ───────────────────────────────────────────────

def test_vocabulary():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web", "python"])
    assert idx.vocabulary == {"python", "web"}
    assert idx.vocabulary_size == 2


# ── get_posting (direct O(1) lookup) ─────────────────────────

def test_get_posting_found():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])

    p = idx.get_posting("python", 1)
    assert p is not None
    assert p.document_id == 1
    assert p.term_frequency == 1


def test_get_posting_not_found():
    idx = InvertedIndex()
    assert idx.get_posting("python", 999) is None


def test_get_posting_wrong_term():
    idx = InvertedIndex()
    idx.add_document(1, ["python"])
    assert idx.get_posting("web", 1) is None


# ── Validation — consistent ──────────────────────────────────

def test_validate_empty_index():
    idx = InvertedIndex()
    assert idx.validate() == []


def test_validate_valid_index():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web", "python"])
    idx.add_document(2, ["javascript", "web"])

    errors = idx.validate()
    assert errors == []


def test_validate_after_operations():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web"])
    idx.add_document(2, ["python"])
    idx.remove_document(1)
    idx.add_document(3, ["web", "api"])

    assert idx.validate() == []


# ── Validation — detects inconsistency ───────────────────────

def test_validate_detects_tf_mismatch():
    idx = InvertedIndex()
    idx.add_document(1, ["python"])

    # Tamper: corrupt TF
    idx._postings["python"][1].term_frequency = 99

    errors = idx.validate()
    assert len(errors) >= 1
    assert any("TF=" in e for e in errors)


def test_validate_detects_unsorted_positions():
    idx = InvertedIndex()
    idx.add_document(1, ["python", "web", "python"])

    # Tamper: reverse positions
    idx._postings["python"][1].positions = [2, 0]

    errors = idx.validate()
    assert len(errors) >= 1
    assert any("not sorted" in e for e in errors)


# ── Test dataset from the plan ───────────────────────────────

def test_plan_dataset():
    """
    The Phase 4 plan's three-document test dataset:
    Doc 1: 'Python is a programming language.'
    Doc 2: 'Python is popular for web development.'
    Doc 3: 'JavaScript is popular for web development.'
    """
    from app.text_processing import tokenize

    doc1 = tokenize("Python is a programming language.")
    doc2 = tokenize("Python is popular for web development.")
    doc3 = tokenize("JavaScript is popular for web development.")

    idx = InvertedIndex()
    idx.add_document(1, doc1)
    idx.add_document(2, doc2)
    idx.add_document(3, doc3)

    # python in docs 1 and 2
    python_docs = [p.document_id for p in idx.get_postings("python")]
    assert python_docs == [1, 2]

    # web in docs 2 and 3
    web_docs = [p.document_id for p in idx.get_postings("web")]
    assert web_docs == [2, 3]

    # development in docs 2 and 3
    dev_docs = [p.document_id for p in idx.get_postings("development")]
    assert dev_docs == [2, 3]

    # javascript only in doc 3
    js_docs = [p.document_id for p in idx.get_postings("javascript")]
    assert js_docs == [3]

    assert idx.get_total_documents() == 3
    assert idx.validate() == []
