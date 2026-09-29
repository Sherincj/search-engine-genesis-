"""
Evaluation reporter — formats evaluation results for display and export.

Responsibilities:
- Human-readable terminal output
- JSON machine-readable output
- Baseline comparison output
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import EvaluationReport


# Metrics to show in the summary report.
SUMMARY_METRICS = (
    "precision@1",
    "precision@3",
    "precision@5",
    "recall@5",
    "f1@5",
    "mrr",
    "map",
    "ndcg@5",
    "ndcg@10",
)


def format_report(report: EvaluationReport, *, verbose: bool = False) -> str:
    """Format an evaluation report as human-readable text."""
    lines: list[str] = []

    lines.append("Search Quality Evaluation")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Queries: {len(report.query_results)}")
    lines.append("")

    # Aggregate metrics.
    lines.append("Aggregate Metrics")
    lines.append("-" * 50)
    for key in SUMMARY_METRICS:
        value = report.aggregate_metrics.get(key, 0.0)
        lines.append(f"  {key:<16s}: {value:.4f}")
    lines.append("")

    # Per-category metrics.
    if report.category_metrics:
        lines.append("Per-Category Metrics")
        lines.append("-" * 50)
        for cat, metrics in sorted(report.category_metrics.items()):
            mrr = metrics.get("mrr", 0.0)
            ndcg5 = metrics.get("ndcg@5", 0.0)
            count = sum(1 for qr in report.query_results if qr.category == cat)
            lines.append(f"  {cat:<20s}  (n={count})  MRR={mrr:.4f}  NDCG@5={ndcg5:.4f}")
        lines.append("")

    # Per-query detail (verbose).
    if verbose:
        lines.append("Per-Query Results")
        lines.append("-" * 50)
        for qr in report.query_results:
            lines.append(f"  Query: {qr.query!r}")
            lines.append(f"  Category: {qr.category}")
            lines.append(f"  Retrieved: {list(qr.retrieved_ids)}")
            mrr = qr.metrics.get("mrr", 0.0)
            ndcg5 = qr.metrics.get("ndcg@5", 0.0)
            ap = qr.metrics.get("ap", 0.0)
            lines.append(f"  MRR={mrr:.4f}  NDCG@5={ndcg5:.4f}  AP={ap:.4f}")
            lines.append("")

    return "\n".join(lines)


def format_json(report: EvaluationReport) -> str:
    """Format an evaluation report as JSON."""
    data = {
        "queries": len(report.query_results),
        "metrics": {
            key: round(report.aggregate_metrics.get(key, 0.0), 4)
            for key in SUMMARY_METRICS
        },
        "category_metrics": {
            cat: {
                key: round(metrics.get(key, 0.0), 4)
                for key in SUMMARY_METRICS
                if key in metrics
            }
            for cat, metrics in report.category_metrics.items()
        },
        "configuration": report.configuration,
        "per_query": [
            {
                "query_id": qr.query_id,
                "query": qr.query,
                "category": qr.category,
                "retrieved": list(qr.retrieved_ids),
                "metrics": {k: round(v, 4) for k, v in qr.metrics.items()},
            }
            for qr in report.query_results
        ],
    }
    return json.dumps(data, indent=2)


def save_baseline(report: EvaluationReport, path: Path) -> None:
    """Save baseline metrics to a JSON file."""
    data = {
        "version": 1,
        "phase": 13,
        "configuration": report.configuration,
        "metrics": {
            key: round(report.aggregate_metrics.get(key, 0.0), 6)
            for key in SUMMARY_METRICS
        },
        "category_metrics": {
            cat: {
                key: round(metrics.get(key, 0.0), 6)
                for key in SUMMARY_METRICS
                if key in metrics
            }
            for cat, metrics in report.category_metrics.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_baseline(path: Path) -> dict:
    """Load a baseline JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_comparison(
    report: EvaluationReport,
    baseline: dict,
) -> str:
    """Format a comparison between current results and a baseline."""
    lines: list[str] = []
    lines.append("Baseline Comparison")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'Metric':<16s}  {'Baseline':>10s}  {'Current':>10s}  {'Delta':>10s}")
    lines.append("-" * 60)

    baseline_metrics = baseline.get("metrics", {})

    for key in SUMMARY_METRICS:
        bl = baseline_metrics.get(key, 0.0)
        cur = report.aggregate_metrics.get(key, 0.0)
        delta = cur - bl
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {key:<16s}  {bl:>10.4f}  {cur:>10.4f}  {sign}{delta:>9.4f}"
        )

    lines.append("")
    return "\n".join(lines)
