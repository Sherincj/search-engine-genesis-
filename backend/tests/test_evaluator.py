"""Tests for the evaluator — uses a fake SearchProvider."""

from evaluation.evaluator import evaluate
from evaluation.models import EvaluationDataset, EvaluationQuery


# ── Fake search provider ─────────────────────────────────────

class FakeSearchProvider:
    """Returns canned results for testing."""

    def __init__(self, results: dict[str, list[int]]):
        self._results = results

    def search(self, query: str, limit: int) -> list[int]:
        return self._results.get(query, [])[:limit]


# ── Helpers ──────────────────────────────────────────────────

def _make_dataset(queries: list[EvaluationQuery]) -> EvaluationDataset:
    return EvaluationDataset(queries=tuple(queries), documents=())


# ── Tests ────────────────────────────────────────────────────

def test_evaluator_basic():
    """Single query with known results and judgments."""
    dataset = _make_dataset([
        EvaluationQuery(
            query_id="q1",
            query="python",
            category="single_term",
            judgments={1: 3, 2: 2, 3: 0},
        ),
    ])
    provider = FakeSearchProvider({"python": [1, 3, 2]})
    report = evaluate(dataset, provider, k_values=(1, 3, 5))

    assert len(report.query_results) == 1
    qr = report.query_results[0]
    assert qr.query_id == "q1"
    assert qr.retrieved_ids == (1, 3, 2)

    # P@1 = 1/1 (doc 1 is relevant)
    assert qr.metrics["precision@1"] == 1.0
    # P@3 = 2/3 (docs 1 and 2 are relevant, 3 is not)
    assert abs(qr.metrics["precision@3"] - 2/3) < 1e-9
    # MRR = 1.0 (first result is relevant)
    assert qr.metrics["mrr"] == 1.0


def test_evaluator_multiple_queries():
    """Multiple queries aggregate correctly."""
    dataset = _make_dataset([
        EvaluationQuery(
            query_id="q1", query="python", category="single_term",
            judgments={1: 3},
        ),
        EvaluationQuery(
            query_id="q2", query="missing", category="zero_result",
            judgments={},
        ),
    ])
    provider = FakeSearchProvider({
        "python": [1, 2, 3],
        "missing": [],
    })
    report = evaluate(dataset, provider, k_values=(1, 3))

    assert len(report.query_results) == 2
    # Aggregate MRR = (1.0 + 0.0) / 2 = 0.5
    assert abs(report.aggregate_metrics["mrr"] - 0.5) < 1e-9


def test_evaluator_category_metrics():
    """Category-level metrics are computed."""
    dataset = _make_dataset([
        EvaluationQuery(
            query_id="q1", query="python", category="single_term",
            judgments={1: 3},
        ),
        EvaluationQuery(
            query_id="q2", query="java", category="single_term",
            judgments={2: 3},
        ),
        EvaluationQuery(
            query_id="q3", query="\"web dev\"", category="phrase",
            judgments={3: 3},
        ),
    ])
    provider = FakeSearchProvider({
        "python": [1],
        "java": [2],
        "\"web dev\"": [99],
    })
    report = evaluate(dataset, provider, k_values=(1,))

    assert "single_term" in report.category_metrics
    assert "phrase" in report.category_metrics
    # Both single_term queries hit: MRR=1.0 each → category MRR=1.0
    assert report.category_metrics["single_term"]["mrr"] == 1.0
    # Phrase query misses: MRR=0.0
    assert report.category_metrics["phrase"]["mrr"] == 0.0


def test_evaluator_empty_dataset():
    """Empty dataset produces empty report."""
    dataset = _make_dataset([])
    provider = FakeSearchProvider({})
    report = evaluate(dataset, provider)

    assert len(report.query_results) == 0
    assert report.aggregate_metrics == {}


def test_evaluator_ndcg_graded():
    """NDCG uses graded relevance from judgments."""
    dataset = _make_dataset([
        EvaluationQuery(
            query_id="q1", query="test", category="single_term",
            judgments={1: 3, 2: 2, 3: 1},
        ),
    ])
    # Perfect order
    provider = FakeSearchProvider({"test": [1, 2, 3]})
    report = evaluate(dataset, provider, k_values=(3,))
    assert report.query_results[0].metrics["ndcg@3"] == 1.0

    # Reverse order
    provider2 = FakeSearchProvider({"test": [3, 2, 1]})
    report2 = evaluate(dataset, provider2, k_values=(3,))
    assert 0 < report2.query_results[0].metrics["ndcg@3"] < 1.0


def test_evaluator_map():
    """MAP is correctly computed as mean of AP values."""
    dataset = _make_dataset([
        EvaluationQuery(
            query_id="q1", query="a", category="single_term",
            judgments={1: 3},
        ),
        EvaluationQuery(
            query_id="q2", query="b", category="single_term",
            judgments={2: 3},
        ),
    ])
    # q1: AP=1.0 (relevant doc at rank 1)
    # q2: AP=0.5 (relevant doc at rank 2)
    provider = FakeSearchProvider({"a": [1, 2], "b": [1, 2]})
    report = evaluate(dataset, provider, k_values=(3,))
    assert abs(report.aggregate_metrics["map"] - 0.75) < 1e-9


def test_evaluator_configuration_recorded():
    """Configuration dict is passed through to report."""
    dataset = _make_dataset([])
    provider = FakeSearchProvider({})
    config = {"bm25_k1": 1.2, "bm25_b": 0.75}
    report = evaluate(dataset, provider, configuration=config)
    assert report.configuration == config
