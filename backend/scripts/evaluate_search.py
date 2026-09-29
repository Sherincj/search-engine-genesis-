"""
Search quality evaluation CLI.

Usage:
    python -m scripts.evaluate_search
    python -m scripts.evaluate_search --json
    python -m scripts.evaluate_search --verbose
    python -m scripts.evaluate_search --save-baseline
    python -m scripts.evaluate_search --compare evaluation/baseline.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the backend directory is on sys.path.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config.settings import settings
from app.indexing import DocumentIndexer, FieldIndex
from app.search.search_engine import SearchEngine
from evaluation.dataset import load_dataset
from evaluation.evaluator import SearchProvider, evaluate
from evaluation.reporter import (
    format_comparison,
    format_json,
    format_report,
    load_baseline,
    save_baseline,
)


class EngineSearchProvider:
    """Adapts SearchEngine to the SearchProvider protocol."""

    def __init__(self, engine: SearchEngine) -> None:
        self._engine = engine

    def search(self, query: str, limit: int) -> list[int]:
        result = self._engine.search(query, limit=limit)
        return [r.document_id for r in result.ranked]


def build_evaluation_index(documents: tuple[dict, ...]) -> FieldIndex:
    """Build a FieldIndex from evaluation documents."""
    index = FieldIndex()
    indexer = DocumentIndexer(index)
    for doc in documents:
        indexer.index_document(
            doc["id"],
            doc["text"],
            title=doc.get("title", ""),
        )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Search quality evaluation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Show per-query results")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as baseline",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Compare with a baseline JSON file",
    )
    args = parser.parse_args()

    # Paths.
    eval_data = _BACKEND_DIR / "evaluation_data"
    docs_path = eval_data / "documents.json"
    queries_path = eval_data / "search_quality.json"

    # Load dataset.
    dataset = load_dataset(docs_path, queries_path)

    # Build isolated index.
    index = build_evaluation_index(dataset.documents)
    engine = SearchEngine(index)
    provider = EngineSearchProvider(engine)

    # Record configuration.
    configuration = {
        "bm25_k1": settings.BM25_K1,
        "bm25_b": settings.BM25_B,
        "title_weight": settings.BM25_TITLE_WEIGHT,
        "body_weight": settings.BM25_BODY_WEIGHT,
        "phrase_boost": settings.PHRASE_BOOST,
        "query_max_weight": settings.QUERY_MAX_WEIGHT,
    }

    # Run evaluation.
    report = evaluate(dataset, provider, configuration=configuration)

    # Output.
    if args.json:
        print(format_json(report))
    else:
        print(format_report(report, verbose=args.verbose))

    # Comparison.
    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.is_absolute():
            baseline_path = _BACKEND_DIR / baseline_path
        baseline = load_baseline(baseline_path)
        print(format_comparison(report, baseline))

    # Save baseline.
    if args.save_baseline:
        baseline_path = _BACKEND_DIR / "evaluation" / "baseline.json"
        save_baseline(report, baseline_path)
        print(f"Baseline saved to {baseline_path}")


if __name__ == "__main__":
    main()
