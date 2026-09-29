"""
Evaluation domain models — immutable data structures for search evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


VALID_CATEGORIES = frozenset({
    "single_term",
    "multi_term",
    "phrase",
    "mixed",
    "repeated_term",
    "zero_result",
})

VALID_RELEVANCE_GRADES = frozenset({0, 1, 2, 3})


@dataclass(frozen=True)
class EvaluationQuery:
    """A single evaluation query with relevance judgments."""
    query_id: str
    query: str
    category: str
    judgments: dict[int, int]  # doc_id -> relevance grade (0-3)


@dataclass(frozen=True)
class QueryEvaluation:
    """Per-query evaluation result."""
    query_id: str
    query: str
    category: str
    retrieved_ids: tuple[int, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvaluationReport:
    """Complete evaluation report."""
    query_results: tuple[QueryEvaluation, ...]
    aggregate_metrics: dict[str, float]
    category_metrics: dict[str, dict[str, float]]
    configuration: dict[str, float]


@dataclass(frozen=True)
class EvaluationDataset:
    """Loaded evaluation dataset."""
    queries: tuple[EvaluationQuery, ...]
    documents: tuple[dict, ...]  # raw document dicts
