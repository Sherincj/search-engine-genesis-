"""Tests for candidate merger."""

from app.hybrid_search.candidate_merger import merge_candidates


def test_merge_overlapping():
    lex = {1: 10.0, 2: 8.0, 3: 5.0}
    sem = {2: 0.8, 3: 0.7, 4: 0.9}
    result = merge_candidates(lex, sem)
    ids = {c.document_id for c in result}
    assert ids == {1, 2, 3, 4}


def test_merge_missing_semantic():
    """Document in BM25 only → semantic = 0.0."""
    lex = {1: 10.0}
    sem = {2: 0.8}
    result = merge_candidates(lex, sem)
    doc1 = next(c for c in result if c.document_id == 1)
    assert doc1.lexical_score == 10.0
    assert doc1.semantic_score == 0.0


def test_merge_missing_lexical():
    """Document in semantic only → BM25 = 0.0."""
    lex = {1: 10.0}
    sem = {2: 0.8}
    result = merge_candidates(lex, sem)
    doc2 = next(c for c in result if c.document_id == 2)
    assert doc2.lexical_score == 0.0
    assert doc2.semantic_score == 0.8


def test_merge_both_present():
    lex = {1: 10.0}
    sem = {1: 0.9}
    result = merge_candidates(lex, sem)
    assert len(result) == 1
    assert result[0].lexical_score == 10.0
    assert result[0].semantic_score == 0.9


def test_merge_completely_different():
    lex = {1: 5.0, 2: 3.0}
    sem = {3: 0.9, 4: 0.8}
    result = merge_candidates(lex, sem)
    assert len(result) == 4


def test_merge_one_empty_lexical():
    result = merge_candidates({}, {1: 0.9, 2: 0.8})
    assert len(result) == 2
    for c in result:
        assert c.lexical_score == 0.0


def test_merge_one_empty_semantic():
    result = merge_candidates({1: 10.0, 2: 8.0}, {})
    assert len(result) == 2
    for c in result:
        assert c.semantic_score == 0.0


def test_merge_both_empty():
    assert merge_candidates({}, {}) == []


def test_merge_no_duplicates():
    lex = {1: 10.0, 2: 8.0}
    sem = {1: 0.9, 2: 0.7}
    result = merge_candidates(lex, sem)
    assert len(result) == 2


def test_merge_preserves_scores():
    lex = {5: 12.3}
    sem = {5: 0.456}
    result = merge_candidates(lex, sem)
    assert result[0].lexical_score == 12.3
    assert result[0].semantic_score == 0.456
