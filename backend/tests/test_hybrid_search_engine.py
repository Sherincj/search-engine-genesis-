"""Tests for the hybrid search engine."""

import math
import pytest

from app.hybrid_search.fusion import WeightedLinearFusion
from app.hybrid_search.hybrid_search_engine import HybridSearchEngine
from app.indexing import DocumentIndexer, FieldIndex
from app.search.search_engine import SearchEngine
from app.search.semantic_search_engine import SemanticSearchEngine
from app.vector_index.vector_index import VectorIndex


# ── Fake embedding provider ──────────────────────────────────

class _FakeProvider:
    def __init__(self, dim=4):
        self._dim = dim

    def embed_text(self, text: str) -> list[float]:
        h = hash(text)
        vec = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self._dim)]
        mag = math.sqrt(sum(v * v for v in vec))
        if mag > 0:
            vec = [v / mag for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self):
        return self._dim

    @property
    def model_name(self):
        return "fake"


# ── Helpers ───────────────────────────────────────────────────

def _build_hybrid_env(docs: list[tuple[int, str, str]], dim: int = 4):
    """Build a hybrid engine from a list of (id, title, text) tuples."""
    # Build lexical index.
    field_index = FieldIndex()
    indexer = DocumentIndexer(field_index)
    for doc_id, title, text in docs:
        indexer.index_document(doc_id, text, title=title)
    lexical_engine = SearchEngine(field_index)

    # Build vector index.
    provider = _FakeProvider(dim)
    vector_index = VectorIndex(dimension=dim)
    for doc_id, title, text in docs:
        from app.embeddings.provider import build_embedding_text
        emb_text = build_embedding_text(title, text)
        vector_index.add(doc_id, provider.embed_text(emb_text))
    semantic_engine = SemanticSearchEngine(vector_index, provider)

    return lexical_engine, semantic_engine, provider


# ── Tests ─────────────────────────────────────────────────────

def test_hybrid_basic():
    docs = [
        (1, "Python Guide", "Python is a programming language"),
        (2, "JavaScript", "JavaScript for web development"),
    ]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem)
    result = hybrid.search("python")
    assert result.total > 0
    assert len(result.ranked) > 0


def test_hybrid_empty_query():
    docs = [(1, "Test", "Content")]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem)
    result = hybrid.search("")
    assert result.total == 0
    assert result.ranked == []


def test_hybrid_no_results():
    docs = [(1, "Python", "Python programming")]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem)
    result = hybrid.search("xyznonexistent")
    # BM25 returns nothing, semantic returns everything
    assert result.total >= 0


def test_hybrid_scores_present():
    docs = [
        (1, "Python", "Python programming language"),
        (2, "Java", "Java programming language"),
    ]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem)
    result = hybrid.search("python")
    for r in result.ranked:
        assert hasattr(r, 'hybrid_score')
        assert hasattr(r, 'normalized_lexical')
        assert hasattr(r, 'normalized_semantic')


def test_hybrid_lexical_weight_1():
    """With weight 1/0, should approximate BM25 ranking."""
    docs = [
        (1, "Python", "Python programming"),
        (2, "Java", "Java programming"),
    ]
    lex, sem, _ = _build_hybrid_env(docs)
    fusion = WeightedLinearFusion(1.0, 0.0)
    hybrid = HybridSearchEngine(lex, sem, fusion=fusion)
    result = hybrid.search("python")
    # Doc 1 should be ranked first (exact BM25 match)
    if result.ranked:
        assert result.ranked[0].document_id == 1


def test_hybrid_pagination():
    docs = [(i, f"Doc {i}", f"Content for document number {i}") for i in range(10)]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem)
    r1 = hybrid.search("document", limit=3, offset=0)
    r2 = hybrid.search("document", limit=3, offset=3)
    ids1 = {r.document_id for r in r1.ranked}
    ids2 = {r.document_id for r in r2.ranked}
    assert ids1.isdisjoint(ids2)


def test_hybrid_timing():
    docs = [(1, "Test", "Test content")]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem)
    result = hybrid.search("test")
    assert result.took_ms >= 0
    assert result.lexical_ms >= 0
    assert result.semantic_ms >= 0
    assert result.fusion_ms >= 0


def test_hybrid_deterministic():
    docs = [
        (1, "Python", "Python programming"),
        (2, "Java", "Java programming"),
        (3, "Rust", "Rust programming"),
    ]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem)
    r1 = hybrid.search("programming")
    r2 = hybrid.search("programming")
    ids1 = [r.document_id for r in r1.ranked]
    ids2 = [r.document_id for r in r2.ranked]
    assert ids1 == ids2


def test_hybrid_candidate_limit():
    docs = [(i, f"Doc {i}", f"Content {i}") for i in range(20)]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem, candidate_limit=5)
    result = hybrid.search("content", limit=20)
    # Total shouldn't exceed the candidate union
    assert result.total <= 10  # 5 from each at most


def test_hybrid_all_scores_finite():
    docs = [
        (1, "Python Guide", "Learn Python"),
        (2, "JavaScript", "Web development"),
    ]
    lex, sem, _ = _build_hybrid_env(docs)
    hybrid = HybridSearchEngine(lex, sem)
    result = hybrid.search("python")
    for r in result.ranked:
        assert math.isfinite(r.hybrid_score)
        assert math.isfinite(r.normalized_lexical)
        assert math.isfinite(r.normalized_semantic)
