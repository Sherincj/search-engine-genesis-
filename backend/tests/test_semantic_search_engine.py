"""Tests for the semantic search engine using a fake embedding provider."""

import hashlib
import math
import pytest

from app.search.semantic_search_engine import SemanticSearchEngine
from app.vector_index.vector_index import VectorIndex


# ── Fake provider (reuse from test_embedding_provider) ────────

class FakeProvider:
    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self._model_name = "fake-model"

    def embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]
        mag = math.sqrt(sum(v * v for v in vec))
        if mag > 0:
            vec = [v / mag for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


# ── Helpers ───────────────────────────────────────────────────

def _build_semantic_env(dim: int = 4) -> tuple[VectorIndex, FakeProvider, SemanticSearchEngine]:
    provider = FakeProvider(dimension=dim)
    index = VectorIndex(dimension=dim)
    engine = SemanticSearchEngine(index, provider)
    return index, provider, engine


# ── Tests ─────────────────────────────────────────────────────

def test_semantic_search_basic():
    index, provider, engine = _build_semantic_env()
    # Add a document whose vector is the embedding of "python programming"
    vec = provider.embed_text("python programming")
    index.add(1, vec)

    result = engine.search("python programming", limit=5)
    assert result.total == 1
    assert result.ranked[0].document_id == 1
    assert result.ranked[0].score == pytest.approx(1.0)  # identical vectors


def test_semantic_search_rejects_provider_dimension_mismatch():
    index = VectorIndex(dimension=3)
    with pytest.raises(ValueError, match="dimension"):
        SemanticSearchEngine(index, FakeProvider(dimension=4))


def test_semantic_search_empty_query():
    _, _, engine = _build_semantic_env()
    result = engine.search("")
    assert result.total == 0
    assert result.ranked == []


def test_semantic_search_empty_index():
    _, _, engine = _build_semantic_env()
    result = engine.search("python", limit=5)
    assert result.total == 0
    assert result.ranked == []


def test_semantic_search_ordering():
    index, provider, engine = _build_semantic_env()
    # Add docs with different embeddings
    for doc_id, text in [(1, "python programming"), (2, "javascript web"), (3, "database sql")]:
        index.add(doc_id, provider.embed_text(text))

    result = engine.search("python programming", limit=3)
    assert result.total == 3
    # First result should be most similar (doc 1 = exact match)
    assert result.ranked[0].document_id == 1
    assert result.ranked[0].score == pytest.approx(1.0)


def test_semantic_search_top_k():
    index, provider, engine = _build_semantic_env()
    for i in range(10):
        index.add(i, provider.embed_text(f"document {i}"))

    result = engine.search("document 5", limit=3)
    assert len(result.ranked) == 3


def test_semantic_search_pagination():
    index, provider, engine = _build_semantic_env()
    for i in range(5):
        index.add(i, provider.embed_text(f"doc {i}"))

    r1 = engine.search("doc 0", limit=2, offset=0)
    r2 = engine.search("doc 0", limit=2, offset=2)

    ids1 = {r.document_id for r in r1.ranked}
    ids2 = {r.document_id for r in r2.ranked}
    assert ids1.isdisjoint(ids2)  # no overlap
    assert r1.total == 5
    assert r2.total == 5


def test_semantic_search_timing():
    _, _, engine = _build_semantic_env()
    result = engine.search("test query")
    assert result.took_ms >= 0
    assert result.embedding_ms >= 0
    assert result.search_ms >= 0


def test_semantic_search_score_is_cosine():
    """Score should be cosine similarity, not BM25."""
    index, provider, engine = _build_semantic_env()
    vec = provider.embed_text("test")
    index.add(1, vec)
    result = engine.search("test", limit=1)
    # Identical vectors → cosine = 1.0
    assert result.ranked[0].score == pytest.approx(1.0)


def test_semantic_search_deterministic():
    index, provider, engine = _build_semantic_env()
    for i in range(5):
        index.add(i, provider.embed_text(f"item {i}"))

    r1 = engine.search("item 3", limit=5)
    r2 = engine.search("item 3", limit=5)
    assert [r.document_id for r in r1.ranked] == [r.document_id for r in r2.ranked]
    assert [r.score for r in r1.ranked] == [r.score for r in r2.ranked]
