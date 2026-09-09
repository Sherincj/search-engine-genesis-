"""
TF-IDF ranker — scores candidate documents using TF-IDF cosine similarity.

Pipeline::

    query terms
    + candidate IDs
    + inverted index
        ↓
    TF-IDF vectors
        ↓
    cosine similarity
        ↓
    ranked results

This module does NOT access SQLite, FastAPI, or HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.indexing.inverted_index import InvertedIndex
from app.search.tfidf import (
    cosine_similarity,
    inverse_document_frequency,
    term_frequency,
    tf_idf,
)


@dataclass
class RankedCandidate:
    """A candidate document with its TF-IDF cosine similarity score."""

    document_id: int
    score: float
    matched_terms: set[str] = field(default_factory=set)


def rank_candidates(
    query_terms: list[str],
    candidate_doc_ids: list[int],
    index: InvertedIndex,
) -> list[RankedCandidate]:
    """
    Rank candidate documents by TF-IDF cosine similarity.

    Parameters
    ----------
    query_terms : list[str]
        Normalised query tokens (from query parser).
    candidate_doc_ids : list[int]
        Document IDs returned by candidate retrieval.
    index : InvertedIndex
        The inverted index (source of TF, DF, N, document terms).

    Returns
    -------
    list[RankedCandidate]
        Candidates sorted by:
        1. Cosine similarity — descending
        2. Document ID — ascending (tie-breaker)
    """
    if not query_terms or not candidate_doc_ids:
        return []

    total_docs = index.get_total_documents()
    if total_docs == 0:
        return []

    # Deduplicate query terms.
    unique_query_terms = set(query_terms)

    # ── Build query vector ───────────────────────────────────
    # Query TF is always 1 (each unique term appears once),
    # so query weight = IDF(term).
    query_vector: dict[str, float] = {}
    for term in unique_query_terms:
        df = index.get_document_frequency(term)
        if df == 0:
            continue  # Term not in index — skip.
        idf = inverse_document_frequency(total_docs, df)
        # Query TF = 1 + log(1) = 1, so weight = 1 * IDF = IDF.
        query_vector[term] = term_frequency(1) * idf

    if not query_vector:
        return []

    # ── Score each candidate ─────────────────────────────────
    results: list[RankedCandidate] = []

    for doc_id in candidate_doc_ids:
        doc_vector = _build_document_vector(doc_id, index, total_docs)

        score = cosine_similarity(query_vector, doc_vector)

        # Which query terms matched this document?
        matched = unique_query_terms & set(
            index.get_document_terms(doc_id)
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


def _build_document_vector(
    doc_id: int,
    index: InvertedIndex,
    total_docs: int,
) -> dict[str, float]:
    """
    Build the full TF-IDF vector for a document.

    Uses *all* terms in the document (not just query terms) so that
    the document magnitude is correct for cosine similarity.
    """
    doc_terms = index.get_document_terms(doc_id)
    vector: dict[str, float] = {}

    for term in doc_terms:
        posting = index.get_posting(term, doc_id)
        if posting is None:
            continue

        df = index.get_document_frequency(term)
        if df == 0:
            continue

        weight = tf_idf(posting.term_frequency, total_docs, df)
        vector[term] = weight

    return vector
