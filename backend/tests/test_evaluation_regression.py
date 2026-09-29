"""
Regression tests for the evaluation framework.

Ensures:
- Baseline dataset loads successfully
- Evaluation completes without errors
- All required metrics exist
- Values are finite
- Results are deterministic
"""

from pathlib import Path

from app.indexing import DocumentIndexer, FieldIndex
from app.search.search_engine import SearchEngine
from evaluation.dataset import load_dataset
from evaluation.evaluator import evaluate
from evaluation.reporter import SUMMARY_METRICS


_EVAL_DATA = Path(__file__).resolve().parent.parent / "evaluation_data"
_DOCS_PATH = _EVAL_DATA / "documents.json"
_QUERIES_PATH = _EVAL_DATA / "search_quality.json"


class _EngineProvider:
    def __init__(self, engine: SearchEngine):
        self._engine = engine

    def search(self, query: str, limit: int) -> list[int]:
        result = self._engine.search(query, limit=limit)
        return [r.document_id for r in result.ranked]


def _build_eval_env():
    dataset = load_dataset(_DOCS_PATH, _QUERIES_PATH)
    index = FieldIndex()
    indexer = DocumentIndexer(index)
    for doc in dataset.documents:
        indexer.index_document(doc["id"], doc["text"], title=doc.get("title", ""))
    engine = SearchEngine(index)
    provider = _EngineProvider(engine)
    return dataset, provider


def test_dataset_loads():
    """Baseline dataset loads without errors."""
    dataset = load_dataset(_DOCS_PATH, _QUERIES_PATH)
    assert len(dataset.queries) >= 15
    assert len(dataset.documents) >= 20


def test_evaluation_completes():
    """Evaluation runs to completion."""
    dataset, provider = _build_eval_env()
    report = evaluate(dataset, provider)
    assert len(report.query_results) == len(dataset.queries)


def test_all_metrics_exist():
    """All summary metrics are present in the report."""
    dataset, provider = _build_eval_env()
    report = evaluate(dataset, provider)
    for key in SUMMARY_METRICS:
        assert key in report.aggregate_metrics, f"Missing metric: {key}"


def test_metrics_are_finite():
    """All metric values are finite floats (no NaN or Inf)."""
    dataset, provider = _build_eval_env()
    report = evaluate(dataset, provider)
    import math
    for key, value in report.aggregate_metrics.items():
        assert math.isfinite(value), f"{key} = {value} is not finite"


def test_metrics_in_range():
    """All metric values are between 0 and 1."""
    dataset, provider = _build_eval_env()
    report = evaluate(dataset, provider)
    for key, value in report.aggregate_metrics.items():
        assert 0.0 <= value <= 1.0, f"{key} = {value} not in [0, 1]"


def test_deterministic():
    """Running evaluation twice produces identical results."""
    dataset, provider = _build_eval_env()
    r1 = evaluate(dataset, provider)
    r2 = evaluate(dataset, provider)

    for key in SUMMARY_METRICS:
        assert r1.aggregate_metrics[key] == r2.aggregate_metrics[key], (
            f"{key}: {r1.aggregate_metrics[key]} != {r2.aggregate_metrics[key]}"
        )


def test_per_query_results_complete():
    """Each query produces a QueryEvaluation with metrics."""
    dataset, provider = _build_eval_env()
    report = evaluate(dataset, provider)

    for qr in report.query_results:
        assert qr.query_id
        assert qr.query
        assert qr.category
        assert isinstance(qr.retrieved_ids, tuple)
        assert "mrr" in qr.metrics
        assert "ap" in qr.metrics


def test_category_metrics_present():
    """Category-level metrics are present for all categories."""
    dataset, provider = _build_eval_env()
    report = evaluate(dataset, provider)

    categories = {q.category for q in dataset.queries}
    for cat in categories:
        assert cat in report.category_metrics, f"Missing category: {cat}"
