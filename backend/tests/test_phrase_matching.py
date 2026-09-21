"""
Tests for the phrase matching module.

Verifies exact adjacency detection using positional data from the
inverted index.
"""

from app.indexing.field_index import FieldIndex
from app.search.phrase_matching import (
    PhraseMatch,
    check_phrase_in_field,
    find_phrase_matches,
    find_phrase_positions,
    get_phrase_candidates,
)


# ── find_phrase_positions ────────────────────────────────────

def test_two_term_phrase_match():
    """python at 2, development at 3 → match at 2."""
    result = find_phrase_positions([[2, 8, 15], [3, 10, 20]])
    assert result == [2]


def test_two_term_phrase_no_match():
    """python at 2, development at 4 → no match (gap)."""
    result = find_phrase_positions([[2, 8, 15], [4, 10, 20]])
    assert result == []


def test_wrong_ordering():
    """python at 5, development at 4 → no match (reversed)."""
    result = find_phrase_positions([[5], [4]])
    assert result == []


def test_three_term_phrase():
    """python at 4, web at 5, development at 6 → match at 4."""
    result = find_phrase_positions([[4], [5], [6]])
    assert result == [4]


def test_three_term_phrase_gap():
    """python at 4, web at 6, development at 7 → no match."""
    result = find_phrase_positions([[4], [6], [7]])
    assert result == []


def test_multiple_match_positions():
    """Phrase matches at multiple positions."""
    # "python development" at positions 2 and 10.
    result = find_phrase_positions([[2, 10], [3, 11]])
    assert result == [2, 10]


def test_single_term_phrase():
    """A single-term 'phrase' returns all positions."""
    result = find_phrase_positions([[2, 5, 10]])
    assert result == [2, 5, 10]


def test_empty_positions():
    """No positions → no match."""
    result = find_phrase_positions([[]])
    assert result == []


def test_missing_term():
    """Empty list of position lists → no match."""
    result = find_phrase_positions([])
    assert result == []


def test_one_term_missing_positions():
    """One term has positions, other doesn't → no match."""
    result = find_phrase_positions([[2, 5], []])
    assert result == []


def test_repeated_positions():
    """Same position in multiple terms shouldn't cause issues."""
    result = find_phrase_positions([[5, 5], [6]])
    assert 5 in result


# ── check_phrase_in_field ────────────────────────────────────

def _build_index_with_doc(doc_id, tokens):
    """Build a FieldIndex with a single document in body."""
    idx = FieldIndex()
    idx.add_document(doc_id, body_tokens=tokens)
    return idx


def test_phrase_in_body():
    idx = _build_index_with_doc(1, ["python", "development", "guide"])
    positions = check_phrase_in_field(
        ["python", "development"], 1, idx.body_index
    )
    assert positions == [0]


def test_phrase_not_in_body():
    idx = _build_index_with_doc(1, ["python", "guide", "development"])
    positions = check_phrase_in_field(
        ["python", "development"], 1, idx.body_index
    )
    assert positions == []


def test_phrase_in_title():
    idx = FieldIndex()
    idx.add_document(1, title_tokens=["python", "development"])
    positions = check_phrase_in_field(
        ["python", "development"], 1, idx.title_index
    )
    assert positions == [0]


# ── find_phrase_matches (field-aware) ────────────────────────

def test_phrase_match_in_body_only():
    idx = FieldIndex()
    idx.add_document(
        1,
        title_tokens=["guide"],
        body_tokens=["python", "development", "tutorial"],
    )
    matches = find_phrase_matches(["python", "development"], 1, idx)
    assert len(matches) == 1
    assert matches[0].field == "body"


def test_phrase_match_in_title_only():
    idx = FieldIndex()
    idx.add_document(
        1,
        title_tokens=["python", "development"],
        body_tokens=["some", "other", "text"],
    )
    matches = find_phrase_matches(["python", "development"], 1, idx)
    assert len(matches) == 1
    assert matches[0].field == "title"


def test_phrase_match_in_both_fields():
    idx = FieldIndex()
    idx.add_document(
        1,
        title_tokens=["python", "development"],
        body_tokens=["learn", "python", "development", "today"],
    )
    matches = find_phrase_matches(["python", "development"], 1, idx)
    assert len(matches) == 2
    fields = {m.field for m in matches}
    assert "title" in fields
    assert "body" in fields


def test_phrase_cross_field_no_match():
    """Phrase split across title and body must NOT match."""
    idx = FieldIndex()
    idx.add_document(
        1,
        title_tokens=["python"],
        body_tokens=["development"],
    )
    matches = find_phrase_matches(["python", "development"], 1, idx)
    assert len(matches) == 0


def test_phrase_no_match_anywhere():
    idx = FieldIndex()
    idx.add_document(
        1,
        title_tokens=["web"],
        body_tokens=["programming"],
    )
    matches = find_phrase_matches(["python", "development"], 1, idx)
    assert len(matches) == 0


# ── get_phrase_candidates ────────────────────────────────────

def test_phrase_candidates_basic():
    idx = FieldIndex()
    idx.add_document(1, body_tokens=["python", "development"])
    idx.add_document(2, body_tokens=["python", "guide"])
    idx.add_document(3, body_tokens=["web", "development"])

    candidates = get_phrase_candidates(["python", "development"], idx)
    assert 1 in candidates  # has both terms
    assert 2 not in candidates  # only python
    assert 3 not in candidates  # only development


def test_phrase_candidates_empty_phrase():
    idx = FieldIndex()
    assert get_phrase_candidates([], idx) == set()


def test_phrase_candidates_title_field():
    idx = FieldIndex()
    idx.add_document(1, title_tokens=["python", "development"])
    candidates = get_phrase_candidates(["python", "development"], idx)
    assert 1 in candidates
