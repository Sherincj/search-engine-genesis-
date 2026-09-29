"""
Evaluator — runs search queries against a SearchProvider and calculates metrics.

The evaluator is decoupled from the actual search engine implementation
via the SearchProvider protocol.
"""

from __future__ import annotations

from typing import Protocol

from evaluation.metrics import (
    average_precision,
    f1_at_k,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.models import (
    EvaluationDataset,
    EvaluationQuery,
    EvaluationReport,
    QueryEvaluation,
)


class SearchProvider(Protocol):
    """Abstraction over any search implementation."""

    def search(self, query: str, limit: int) -> list[int]:
        """Execute a search and return ranked document IDs."""
        ...


def evaluate(
    dataset: EvaluationDataset,
    provider: SearchProvider,
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    configuration: dict[str, float] | None = None,
) -> EvaluationReport:
    """
    Run evaluation queries against a search provider.

    Parameters
    ----------
    dataset : EvaluationDataset
        The evaluation dataset with queries and judgments.
    provider : SearchProvider
        The search implementation to evaluate.
    k_values : tuple[int, ...]
        K values for Precision@K, Recall@K, etc.
    configuration : dict[str, float] | None
        Search configuration metadata for the report.

    Returns
    -------
    EvaluationReport
        Per-query and aggregate evaluation results.
    """
    max_k = max(k_values) if k_values else 10

    query_results: list[QueryEvaluation] = []

    for eq in dataset.queries:
        retrieved = provider.search(eq.query, limit=max_k)
        metrics = _compute_query_metrics(retrieved, eq, k_values)
        query_results.append(QueryEvaluation(
            query_id=eq.query_id,
            query=eq.query,
            category=eq.category,
            retrieved_ids=tuple(retrieved),
            metrics=metrics,
        ))

    aggregate = _aggregate_metrics(query_results, k_values)
    category_metrics = _category_metrics(query_results, k_values)

    return EvaluationReport(
        query_results=tuple(query_results),
        aggregate_metrics=aggregate,
        category_metrics=category_metrics,
        configuration=configuration or {},
    )


def _compute_query_metrics(
    retrieved: list[int],
    eq: EvaluationQuery,
    k_values: tuple[int, ...],
) -> dict[str, float]:
    """Compute all metrics for a single query."""
    # Binary relevant set: relevance > 0
    relevant = {doc_id for doc_id, grade in eq.judgments.items() if grade > 0}

    metrics: dict[str, float] = {}

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"f1@{k}"] = f1_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, eq.judgments, k)

    metrics["mrr"] = reciprocal_rank(retrieved, relevant)
    metrics["ap"] = average_precision(retrieved, relevant)

    return metrics


def _aggregate_metrics(
    results: list[QueryEvaluation],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    """Aggregate per-query metrics into means."""
    if not results:
        return {}

    n = len(results)
    aggregate: dict[str, float] = {}

    # Collect all metric keys from first result.
    metric_keys = list(results[0].metrics.keys())

    for key in metric_keys:
        total = sum(qr.metrics.get(key, 0.0) for qr in results)
        aggregate[key] = total / n

    # MAP is just the aggregate AP.
    aggregate["map"] = aggregate.get("ap", 0.0)

    return aggregate


def _category_metrics(
    results: list[QueryEvaluation],
    k_values: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics per query category."""
    categories: dict[str, list[QueryEvaluation]] = {}
    for qr in results:
        categories.setdefault(qr.category, []).append(qr)

    category_agg: dict[str, dict[str, float]] = {}
    for cat, cat_results in sorted(categories.items()):
        category_agg[cat] = _aggregate_metrics(cat_results, k_values)

    return category_agg
