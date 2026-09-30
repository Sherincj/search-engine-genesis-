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
from evaluation.semantic_eval import format_comparison_table


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
    parser.add_argument(
        "--engine",
        type=str,
        choices=["lexical", "semantic", "compare"],
        default="lexical",
        help="Which search engine to evaluate",
    )
    args = parser.parse_args()

    # Paths.
    eval_data = _BACKEND_DIR / "evaluation_data"
    docs_path = eval_data / "documents.json"
    queries_path = eval_data / "search_quality.json"

    # Load dataset.
    dataset = load_dataset(docs_path, queries_path)

    # Build evaluation environments. Comparison deliberately evaluates both
    # independent engines against the same immutable judgments.
    if args.engine in {"semantic", "compare"}:
        from app.embeddings.sentence_transformer import SentenceTransformerProvider
        from evaluation.semantic_eval import SemanticSearchProvider, build_evaluation_environment

        provider = SentenceTransformerProvider(
            model_name=settings.EMBEDDING_MODEL,
            normalize=settings.EMBEDDING_NORMALIZE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
        lexical_engine, semantic_engine = build_evaluation_environment(dataset, provider)
        search_provider = SemanticSearchProvider(semantic_engine)

        configuration = {
            "engine": "semantic",
            "model_name": settings.EMBEDDING_MODEL,
            "normalize": settings.EMBEDDING_NORMALIZE,
        }
    else:
        index = build_evaluation_index(dataset.documents)
        engine = SearchEngine(index)
        search_provider = EngineSearchProvider(engine)

        configuration = {
            "engine": "lexical",
            "bm25_k1": settings.BM25_K1,
            "bm25_b": settings.BM25_B,
            "title_weight": settings.BM25_TITLE_WEIGHT,
            "body_weight": settings.BM25_BODY_WEIGHT,
            "phrase_boost": settings.PHRASE_BOOST,
            "query_max_weight": settings.QUERY_MAX_WEIGHT,
        }

    if args.engine == "compare":
        lexical_report = evaluate(
            dataset,
            EngineSearchProvider(lexical_engine),
            configuration={"engine": "lexical"},
        )
        semantic_report = evaluate(dataset, search_provider, configuration=configuration)
        print(format_comparison_table(
            lexical_report.aggregate_metrics,
            semantic_report.aggregate_metrics,
        ))
        print("Lexical category metrics")
        print(format_report(lexical_report, verbose=args.verbose))
        print("Semantic category metrics")
        print(format_report(semantic_report, verbose=args.verbose))
        return

    # Run a single-engine evaluation.
    report = evaluate(dataset, search_provider, configuration=configuration)

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
