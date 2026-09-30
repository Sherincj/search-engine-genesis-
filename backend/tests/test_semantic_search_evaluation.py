"""
Tests for semantic search evaluation using a fake embedding provider.

Verifies that the evaluation framework correctly evaluates semantic search
without requiring a real embedding model.
"""

import hashlib
import math
from pathlib import Path

from app.search.semantic_search_engine import SemanticSearchEngine
from app.vector_index.vector_index import VectorIndex
from evaluation.dataset import load_dataset
from evaluation.evaluator import evaluate
from evaluation.semantic_eval import SemanticSearchProvider


_EVAL_DATA = Path(__file__).resolve().parent.parent / "evaluation_data"
_DOCS_PATH = _EVAL_DATA / "documents.json"
_QUERIES_PATH = _EVAL_DATA / "search_quality.json"


# ── Fake embedding provider ──────────────────────────────────

class _FakeProvider:
    def __init__(self, dim: int = 8):
        self._dim = dim
        self._model_name = "fake-eval-model"

    def embed_text(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [digest[i % len(digest)] / 255.0 for i in range(self._dim)]
        mag = math.sqrt(sum(v * v for v in vec))
        if mag > 0:
            vec = [v / mag for v in vec]
        return vec


# ── Tests ─────────────────────────────────────────────────────

def test_semantic_evaluation_completes():
    """Semantic evaluation runs to completion with the evaluation dataset."""
    from app.embeddings.provider import build_embedding_text

    dataset = load_dataset(_DOCS_PATH, _QUERIES_PATH)
    provider = _FakeProvider(dim=8)

    # Build vector index from evaluation documents.
    index = VectorIndex(dimension=8)
    for doc in dataset.documents:
        text = build_embedding_text(doc.get("title", ""), doc["text"])
        vec = provider.embed_text(text)
        index.add(doc["id"], vec)

    engine = SemanticSearchEngine(index, provider)
    sem_provider = SemanticSearchProvider(engine)

    report = evaluate(dataset, sem_provider)

    assert len(report.query_results) == len(dataset.queries)
    assert "mrr" in report.aggregate_metrics
    assert "map" in report.aggregate_metrics
    assert "ndcg@5" in report.aggregate_metrics


def test_semantic_metrics_are_finite():
    from app.embeddings.provider import build_embedding_text

    dataset = load_dataset(_DOCS_PATH, _QUERIES_PATH)
    provider = _FakeProvider(dim=8)

    index = VectorIndex(dimension=8)
    for doc in dataset.documents:
        text = build_embedding_text(doc.get("title", ""), doc["text"])
        index.add(doc["id"], provider.embed_text(text))

    engine = SemanticSearchEngine(index, provider)
    sem_provider = SemanticSearchProvider(engine)
    report = evaluate(dataset, sem_provider)

    for key, value in report.aggregate_metrics.items():
        assert math.isfinite(value), f"{key} = {value}"


def test_semantic_evaluation_deterministic():
    from app.embeddings.provider import build_embedding_text

    dataset = load_dataset(_DOCS_PATH, _QUERIES_PATH)
    provider = _FakeProvider(dim=8)

    index = VectorIndex(dimension=8)
    for doc in dataset.documents:
        text = build_embedding_text(doc.get("title", ""), doc["text"])
        index.add(doc["id"], provider.embed_text(text))

    engine = SemanticSearchEngine(index, provider)
    sem_provider = SemanticSearchProvider(engine)

    r1 = evaluate(dataset, sem_provider)
    r2 = evaluate(dataset, sem_provider)

    for key in r1.aggregate_metrics:
        assert r1.aggregate_metrics[key] == r2.aggregate_metrics[key]


def test_semantic_category_metrics():
    from app.embeddings.provider import build_embedding_text

    dataset = load_dataset(_DOCS_PATH, _QUERIES_PATH)
    provider = _FakeProvider(dim=8)

    index = VectorIndex(dimension=8)
    for doc in dataset.documents:
        text = build_embedding_text(doc.get("title", ""), doc["text"])
        index.add(doc["id"], provider.embed_text(text))

    engine = SemanticSearchEngine(index, provider)
    sem_provider = SemanticSearchProvider(engine)
    report = evaluate(dataset, sem_provider)

    categories = {q.category for q in dataset.queries}
    for cat in categories:
        assert cat in report.category_metrics
