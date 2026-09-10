"""
BM25 ranker — scores candidate documents using BM25.

Pipeline::

    query terms
    + candidate IDs
    + inverted index
    + k1, b
        ↓
    BM25 scores
        ↓
    ranked results

This module does NOT access SQLite, FastAPI, or HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.indexing.inverted_index import InvertedIndex
from app.search.bm25 import bm25_idf, bm25_term_score


@dataclass
class RankedCandidate:
    """A candidate document with its relevance score."""

    document_id: int
    score: float
    matched_terms: set[str] = field(default_factory=set)


def rank_candidates(
    query_terms: list[str],
    candidate_doc_ids: list[int],
    index: InvertedIndex,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[RankedCandidate]:
    """
    Rank candidate documents by BM25 relevance score.

    Parameters
    ----------
    query_terms : list[str]
        Normalised query tokens (from query parser).
    candidate_doc_ids : list[int]
        Document IDs returned by candidate retrieval.
    index : InvertedIndex
        The inverted index (source of TF, DF, N, document lengths).
    k1 : float
        Term-frequency saturation parameter.
    b : float
        Document-length normalization parameter.

    Returns
    -------
    list[RankedCandidate]
        Candidates sorted by:
        1. BM25 score — descending
        2. Document ID — ascending (tie-breaker)
    """
    if not query_terms or not candidate_doc_ids:
        return []

    total_docs = index.get_total_documents()
    if total_docs == 0:
        return []

    avgdl = index.get_average_document_length()
    if avgdl <= 0:
        return []

    # Deduplicate query terms.
    unique_query_terms = set(query_terms)

    # Pre-compute IDF for each query term.
    idfs: dict[str, float] = {}
    for term in unique_query_terms:
        df = index.get_document_frequency(term)
        if df == 0:
            continue
        idfs[term] = bm25_idf(total_docs, df)

    if not idfs:
        return []

    # ── Score each candidate ─────────────────────────────────
    results: list[RankedCandidate] = []

    for doc_id in candidate_doc_ids:
        doc_length = index.get_document_length(doc_id)
        score = 0.0
        matched: set[str] = set()

        for term, idf in idfs.items():
            posting = index.get_posting(term, doc_id)
            if posting is None:
                continue

            matched.add(term)
            score += bm25_term_score(
                term_frequency=posting.term_frequency,
                document_length=doc_length,
                average_document_length=avgdl,
                idf=idf,
                k1=k1,
                b=b,
            )

        results.append(
            RankedCandidate(
                document_id=doc_id,
                score=score,
                matched_terms=matched,
            )
        )

    # ── Sort: score desc, then doc_id asc ────────────────────
    results.sort(key=lambda r: (-r.score, r.document_id))

    return results
