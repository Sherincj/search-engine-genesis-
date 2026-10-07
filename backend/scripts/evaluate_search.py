"""
Search quality evaluation CLI.

Usage:
    python -m scripts.evaluate_search
    python -m scripts.evaluate_search --engine semantic
    python -m scripts.evaluate_search --engine hybrid
    python -m scripts.evaluate_search --engine hybrid --lexical-weight 0.75 --semantic-weight 0.25
    python -m scripts.evaluate_search --engine hybrid --weight-sweep
    python -m scripts.evaluate_search --json
    python -m scripts.evaluate_search --verbose
    python -m scripts.evaluate_search --save-baseline
    python -m scripts.evaluate_search --compare evaluation/baseline.json
"""

from __future__ import annotations

import argparse
import json
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
from evaluation.evaluator import evaluate
from evaluation.reporter import (
    format_comparison,
    format_json,
    format_report,
    load_baseline,
    save_baseline,
    SUMMARY_METRICS,
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


def _build_semantic_provider(dataset):
    """Build semantic engine and provider."""
    from app.embeddings.sentence_transformer import SentenceTransformerProvider
    from evaluation.semantic_eval import SemanticSearchProvider, build_evaluation_environment

    provider = SentenceTransformerProvider(
        model_name=settings.EMBEDDING_MODEL,
        normalize=settings.EMBEDDING_NORMALIZE,
    )
    lexical_engine, semantic_engine = build_evaluation_environment(dataset, provider)
    return lexical_engine, semantic_engine, provider


def _build_hybrid_provider(dataset, lexical_weight, semantic_weight):
    """Build hybrid engine and provider."""
    from app.hybrid_search.fusion import WeightedLinearFusion
    from app.hybrid_search.hybrid_search_engine import HybridSearchEngine
    from evaluation.semantic_eval import SemanticSearchProvider

    lexical_engine, semantic_engine, _ = _build_semantic_provider(dataset)

    fusion = WeightedLinearFusion(lexical_weight, semantic_weight)
    hybrid_engine = HybridSearchEngine(
        lexical_engine,
        semantic_engine,
        fusion=fusion,
        candidate_limit=settings.HYBRID_CANDIDATE_LIMIT,
    )

    class HybridSearchProvider:
        def __init__(self, engine):
            self._engine = engine
        def search(self, query: str, limit: int) -> list[int]:
            result = self._engine.search(query, limit=limit)
            return [r.document_id for r in result.ranked]

    return hybrid_engine, HybridSearchProvider(hybrid_engine), lexical_engine, semantic_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Search quality evaluation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Show per-query results")
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save current results as baseline",
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Compare with a baseline JSON file",
    )
    parser.add_argument(
        "--engine", type=str,
        choices=["lexical", "semantic", "hybrid"],
        default="lexical",
        help="Which search engine to evaluate",
    )
    parser.add_argument(
        "--lexical-weight", type=float, default=None,
        help="Lexical weight for hybrid (default: from settings)",
    )
    parser.add_argument(
        "--semantic-weight", type=float, default=None,
        help="Semantic weight for hybrid (default: from settings)",
    )
    parser.add_argument(
        "--weight-sweep", action="store_true",
        help="Run evaluation across multiple weight configurations",
    )
    args = parser.parse_args()

    # Paths.
    eval_data = _BACKEND_DIR / "evaluation_data"
    docs_path = eval_data / "documents.json"
    queries_path = eval_data / "search_quality.json"

    # Load dataset.
    dataset = load_dataset(docs_path, queries_path)

    # Weight sweep mode.
    if args.weight_sweep:
        _run_weight_sweep(dataset, args)
        return

    # Build provider based on engine choice.
    if args.engine == "semantic":
        from evaluation.semantic_eval import SemanticSearchProvider
        _, semantic_engine, _ = _build_semantic_provider(dataset)
        search_provider = SemanticSearchProvider(semantic_engine)
        configuration = {
            "engine": "semantic",
            "model_name": settings.EMBEDDING_MODEL,
        }

    elif args.engine == "hybrid":
        lw = args.lexical_weight if args.lexical_weight is not None else settings.HYBRID_LEXICAL_WEIGHT
        sw = args.semantic_weight if args.semantic_weight is not None else settings.HYBRID_SEMANTIC_WEIGHT
        _, search_provider, _, _ = _build_hybrid_provider(dataset, lw, sw)
        configuration = {
            "engine": "hybrid",
            "lexical_weight": lw,
            "semantic_weight": sw,
            "candidate_limit": settings.HYBRID_CANDIDATE_LIMIT,
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

    # Run evaluation.
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


def _run_weight_sweep(dataset, args):
    """Run evaluation across multiple weight configurations."""
    from evaluation.semantic_eval import SemanticSearchProvider

    sweep_configs = [
        (1.00, 0.00),
        (0.75, 0.25),
        (0.50, 0.50),
        (0.25, 0.75),
        (0.00, 1.00),
    ]

    print("Running hybrid weight sweep...")
    print("=" * 70)
    print()

    results = {}

    for lw, sw in sweep_configs:
        label = f"{lw:.2f} / {sw:.2f}"
        print(f"Evaluating weights: {label}...")

        _, search_provider, _, _ = _build_hybrid_provider(dataset, lw, sw)
        configuration = {
            "engine": "hybrid",
            "lexical_weight": lw,
            "semantic_weight": sw,
        }
        report = evaluate(dataset, search_provider, configuration=configuration)
        results[label] = report.aggregate_metrics

    # Print summary table.
    print()
    header = f"{'Weights':<14s}"
    for key in SUMMARY_METRICS:
        header += f"  {key:>12s}"
    print(header)
    print("-" * len(header))

    for label, metrics in results.items():
        row = f"  {label:<12s}"
        for key in SUMMARY_METRICS:
            row += f"  {metrics.get(key, 0.0):>12.4f}"
        print(row)

    print()

    # Save sweep results.
    sweep_path = _BACKEND_DIR / "evaluation" / "hybrid_weight_sweep.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_data = {
        label: {k: round(v, 6) for k, v in metrics.items()}
        for label, metrics in results.items()
    }
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"Sweep results saved to {sweep_path}")


if __name__ == "__main__":
    main()
