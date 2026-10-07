"""Tests for score fusion."""

import pytest

from app.hybrid_search.fusion import WeightedLinearFusion, FusionStrategy
from app.hybrid_search.types import HybridCandidate


def _make_candidates(*args) -> list[HybridCandidate]:
    return [HybridCandidate(doc_id, lex, sem) for doc_id, lex, sem in args]


# ── WeightedLinearFusion ──────────────────────────────────────

def test_fusion_satisfies_protocol():
    f = WeightedLinearFusion()
    assert isinstance(f, FusionStrategy)


def test_fusion_equal_weights():
    candidates = _make_candidates(
        (1, 10.0, 0.9),
        (2, 5.0, 0.5),
    )
    fusion = WeightedLinearFusion(0.5, 0.5)
    results = fusion.fuse(candidates)
    assert len(results) == 2
    # Both dimensions normalized: doc1 has max lex and max sem
    assert results[0].document_id == 1
    assert results[0].hybrid_score == pytest.approx(1.0)  # 0.5*1 + 0.5*1


def test_fusion_lexical_only():
    """Weight 1/0 → effectively BM25 ranking."""
    candidates = _make_candidates(
        (1, 10.0, 0.1),  # best BM25
        (2, 5.0, 0.9),   # best semantic
    )
    fusion = WeightedLinearFusion(1.0, 0.0)
    results = fusion.fuse(candidates)
    assert results[0].document_id == 1  # BM25 winner


def test_fusion_semantic_only():
    """Weight 0/1 → effectively semantic ranking."""
    candidates = _make_candidates(
        (1, 10.0, 0.1),
        (2, 5.0, 0.9),
    )
    fusion = WeightedLinearFusion(0.0, 1.0)
    results = fusion.fuse(candidates)
    assert results[0].document_id == 2  # Semantic winner


def test_fusion_75_25():
    candidates = _make_candidates(
        (1, 10.0, 0.1),
        (2, 5.0, 0.9),
    )
    fusion = WeightedLinearFusion(0.75, 0.25)
    results = fusion.fuse(candidates)
    # doc1: 0.75*1.0 + 0.25*0.0 = 0.75
    # doc2: 0.75*0.0 + 0.25*1.0 = 0.25
    assert results[0].document_id == 1


def test_fusion_25_75():
    candidates = _make_candidates(
        (1, 10.0, 0.1),
        (2, 5.0, 0.9),
    )
    fusion = WeightedLinearFusion(0.25, 0.75)
    results = fusion.fuse(candidates)
    # doc1: 0.25*1.0 + 0.75*0.0 = 0.25
    # doc2: 0.25*0.0 + 0.75*1.0 = 0.75
    assert results[0].document_id == 2


def test_fusion_empty():
    fusion = WeightedLinearFusion()
    assert fusion.fuse([]) == []


def test_fusion_single_candidate():
    candidates = _make_candidates((1, 10.0, 0.8))
    fusion = WeightedLinearFusion(0.5, 0.5)
    results = fusion.fuse(candidates)
    assert len(results) == 1
    # Single candidate: both normalized to 1.0
    assert results[0].hybrid_score == pytest.approx(1.0)


def test_fusion_invalid_weights():
    with pytest.raises(ValueError):
        WeightedLinearFusion(-0.5, 0.5)
    with pytest.raises(ValueError):
        WeightedLinearFusion(0.5, -0.5)


def test_fusion_deterministic_tiebreak():
    """Equal hybrid scores → sort by document_id ASC."""
    candidates = _make_candidates(
        (5, 10.0, 0.8),
        (2, 10.0, 0.8),
        (8, 10.0, 0.8),
    )
    fusion = WeightedLinearFusion(0.5, 0.5)
    results = fusion.fuse(candidates)
    ids = [r.document_id for r in results]
    assert ids == [2, 5, 8]


def test_fusion_preserves_raw_scores():
    candidates = _make_candidates((1, 12.3, 0.77))
    fusion = WeightedLinearFusion(0.5, 0.5)
    results = fusion.fuse(candidates)
    assert results[0].lexical_score == 12.3
    assert results[0].semantic_score == 0.77


def test_fusion_normalized_scores_present():
    candidates = _make_candidates(
        (1, 10.0, 0.9),
        (2, 0.0, 0.1),
    )
    fusion = WeightedLinearFusion(0.5, 0.5)
    results = fusion.fuse(candidates)
    assert results[0].normalized_lexical == pytest.approx(1.0)
    assert results[0].normalized_semantic == pytest.approx(1.0)
    assert results[1].normalized_lexical == pytest.approx(0.0)
    assert results[1].normalized_semantic == pytest.approx(0.0)


def test_fusion_missing_semantic_gets_zero():
    """Document with semantic=0 (missing) should normalize properly."""
    candidates = _make_candidates(
        (1, 10.0, 0.8),
        (2, 5.0, 0.0),   # Missing semantic
    )
    fusion = WeightedLinearFusion(0.5, 0.5)
    results = fusion.fuse(candidates)
    doc2 = next(r for r in results if r.document_id == 2)
    assert doc2.normalized_semantic == 0.0


def test_fusion_weight_properties():
    f = WeightedLinearFusion(0.3, 0.7)
    assert f.lexical_weight == 0.3
    assert f.semantic_weight == 0.7
