"""Tests for embedding providers."""

import hashlib
import math
import pytest

from app.embeddings.provider import EmbeddingProvider, build_embedding_text


# ── Fake Embedding Provider ──────────────────────────────────

class FakeEmbeddingProvider:
    """
    Deterministic fake embedding provider for testing.

    Maps known texts to fixed vectors. Unknown texts get a hash-based vector.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self._model_name = "fake-model"

    def embed_text(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic hash-based embedding."""
        # hashlib is stable across Python processes; built-in hash() is not.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self._dimension):
            val = digest[i % len(digest)] / 255.0
            vec.append(val)
        # Normalize to unit vector.
        mag = math.sqrt(sum(v * v for v in vec))
        if mag > 0:
            vec = [v / mag for v in vec]
        return vec


# ── EmbeddingProvider Protocol Tests ─────────────────────────

def test_fake_provider_satisfies_protocol():
    provider = FakeEmbeddingProvider()
    assert isinstance(provider, EmbeddingProvider)

def test_fake_provider_dimension():
    provider = FakeEmbeddingProvider(dimension=8)
    assert provider.dimension == 8

def test_fake_provider_model_name():
    provider = FakeEmbeddingProvider()
    assert provider.model_name == "fake-model"

def test_embed_text_returns_correct_dimension():
    provider = FakeEmbeddingProvider(dimension=16)
    vec = provider.embed_text("hello")
    assert len(vec) == 16

def test_embed_text_deterministic():
    provider = FakeEmbeddingProvider()
    v1 = provider.embed_text("python programming")
    v2 = provider.embed_text("python programming")
    assert v1 == v2

def test_embed_text_different_texts():
    provider = FakeEmbeddingProvider()
    v1 = provider.embed_text("python")
    v2 = provider.embed_text("javascript")
    assert v1 != v2

def test_embed_documents_batch():
    provider = FakeEmbeddingProvider()
    texts = ["python", "javascript", "rust"]
    results = provider.embed_documents(texts)
    assert len(results) == 3
    assert all(len(v) == provider.dimension for v in results)

def test_embed_documents_empty():
    provider = FakeEmbeddingProvider()
    assert provider.embed_documents([]) == []

def test_embed_text_normalized():
    provider = FakeEmbeddingProvider()
    vec = provider.embed_text("test")
    magnitude = math.sqrt(sum(v * v for v in vec))
    assert magnitude == pytest.approx(1.0, abs=1e-6)


# ── build_embedding_text ─────────────────────────────────────

def test_build_embedding_text_both():
    result = build_embedding_text("My Title", "My body text")
    assert result == "My Title\n\nMy body text"

def test_build_embedding_text_title_only():
    assert build_embedding_text("Title", "") == "Title"

def test_build_embedding_text_body_only():
    assert build_embedding_text("", "Body text") == "Body text"

def test_build_embedding_text_both_empty():
    assert build_embedding_text("", "") == ""

def test_build_embedding_text_strips_whitespace():
    result = build_embedding_text("  Title  ", "  Body  ")
    assert result == "Title\n\nBody"
