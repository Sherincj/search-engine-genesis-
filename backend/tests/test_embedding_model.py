"""Integration tests for the real sentence-transformer model.

Skipped unless RUN_EMBEDDING_INTEGRATION_TESTS=1 is set.
Requires sentence-transformers and a model download.
"""

import os
import pytest

RUN_INTEGRATION = os.environ.get("RUN_EMBEDDING_INTEGRATION_TESTS", "") == "1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason="RUN_EMBEDDING_INTEGRATION_TESTS not set")
def test_real_model_loads():
    from app.embeddings.sentence_transformer import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    assert provider.dimension == 384


@pytest.mark.skipif(not RUN_INTEGRATION, reason="RUN_EMBEDDING_INTEGRATION_TESTS not set")
def test_real_model_embed_text():
    from app.embeddings.sentence_transformer import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    vec = provider.embed_text("hello world")
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


@pytest.mark.skipif(not RUN_INTEGRATION, reason="RUN_EMBEDDING_INTEGRATION_TESTS not set")
def test_real_model_batch():
    from app.embeddings.sentence_transformer import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    vecs = provider.embed_documents(["python", "javascript"])
    assert len(vecs) == 2
    assert all(len(v) == 384 for v in vecs)


@pytest.mark.skipif(not RUN_INTEGRATION, reason="RUN_EMBEDDING_INTEGRATION_TESTS not set")
def test_real_model_deterministic():
    from app.embeddings.sentence_transformer import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    v1 = provider.embed_text("test text")
    v2 = provider.embed_text("test text")
    for a, b in zip(v1, v2):
        assert abs(a - b) < 1e-6


@pytest.mark.skipif(not RUN_INTEGRATION, reason="RUN_EMBEDDING_INTEGRATION_TESTS not set")
def test_real_model_normalized():
    import math
    from app.embeddings.sentence_transformer import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2", normalize=True)
    vec = provider.embed_text("test normalization")
    magnitude = math.sqrt(sum(v * v for v in vec))
    assert abs(magnitude - 1.0) < 1e-4
