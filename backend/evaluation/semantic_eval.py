"""
Semantic search evaluation — adapter for the Phase 13 evaluator.

Creates a SearchProvider that uses SemanticSearchEngine instead of BM25,
allowing both to be evaluated with the same relevance judgments.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from app.embeddings.provider import build_embedding_text
from app.indexing import DocumentIndexer, FieldIndex
from app.search.search_engine import SearchEngine
from app.search.semantic_search_engine import SemanticSearchEngine
from app.vector_index.vector_index import VectorIndex
from evaluation.evaluator import evaluate
from evaluation.dataset import load_dataset
from evaluation.models import EvaluationDataset
from evaluation.reporter import format_report, SUMMARY_METRICS


class SemanticSearchProvider:
    """Adapts SemanticSearchEngine to the SearchProvider protocol."""

    def __init__(self, engine: SemanticSearchEngine) -> None:
        self._engine = engine

    def search(self, query: str, limit: int) -> list[int]:
        result = self._engine.search(query, limit=limit)
        return [r.document_id for r in result.ranked]


class LexicalSearchProvider:
    """Adapts SearchEngine (BM25) to the SearchProvider protocol."""

    def __init__(self, engine: SearchEngine) -> None:
        self._engine = engine

    def search(self, query: str, limit: int) -> list[int]:
        result = self._engine.search(query, limit=limit)
        return [r.document_id for r in result.ranked]


def build_evaluation_environment(
    dataset: EvaluationDataset,
    embedding_provider,
) -> tuple[SearchEngine, SemanticSearchEngine]:
    """
    Build both lexical and semantic search engines from the evaluation corpus.

    Parameters
    ----------
    dataset : EvaluationDataset
        Loaded evaluation dataset with documents.
    embedding_provider
        An EmbeddingProvider for generating document vectors.

    Returns
    -------
    tuple[SearchEngine, SemanticSearchEngine]
        (lexical_engine, semantic_engine)
    """
    # Build lexical index.
    field_index = FieldIndex()
    indexer = DocumentIndexer(field_index)
    for doc in dataset.documents:
        indexer.index_document(doc["id"], doc["text"], title=doc.get("title", ""))

    lexical_engine = SearchEngine(field_index)

    # Build vector index.
    dim = embedding_provider.dimension
    vector_index = VectorIndex(dimension=dim)

    texts = []
    doc_ids = []
    for doc in dataset.documents:
        text = build_embedding_text(doc.get("title", ""), doc["text"])
        texts.append(text)
        doc_ids.append(doc["id"])

    embeddings = embedding_provider.embed_documents(texts)
    for doc_id, embedding in zip(doc_ids, embeddings):
        vector_index.add(doc_id, embedding)

    semantic_engine = SemanticSearchEngine(vector_index, embedding_provider)

    return lexical_engine, semantic_engine


def format_comparison_table(
    lexical_metrics: dict[str, float],
    semantic_metrics: dict[str, float],
) -> str:
    """Format a side-by-side comparison of BM25 vs semantic metrics."""
    lines = []
    lines.append("BM25 vs Semantic Search")
    lines.append("=" * 55)
    lines.append("")
    lines.append(f"{'Metric':<16s}  {'BM25':>10s}  {'Semantic':>10s}  {'Delta':>10s}")
    lines.append("-" * 55)

    for key in SUMMARY_METRICS:
        bm25 = lexical_metrics.get(key, 0.0)
        sem = semantic_metrics.get(key, 0.0)
        delta = sem - bm25
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {key:<16s}  {bm25:>10.4f}  {sem:>10.4f}  {sign}{delta:>9.4f}"
        )

    lines.append("")
    return "\n".join(lines)
