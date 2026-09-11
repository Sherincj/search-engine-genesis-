"""
BM25 ranker — field-aware scoring of candidate documents.

Phase 8 update: scores each candidate using field-weighted BM25::

    query terms
    + candidate IDs
    + field index
    + k1, b
    + title_weight, body_weight
        ↓
    per-field BM25 scores
        ↓
    weighted combination
        ↓
    ranked results

Each field (title, body) is scored independently with its own
statistics (DF, doc length, average doc length), then combined::

    final_score = title_weight × title_bm25 + body_weight × body_bm25

This module does NOT access SQLite, FastAPI, or HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.indexing.field_index import FIELD_BODY, FIELD_TITLE, FieldIndex
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
    index: FieldIndex,
    *,
    k1: float = 1.2,
    b: float = 0.75,
    title_weight: float = 2.0,
    body_weight: float = 1.0,
) -> list[RankedCandidate]:
    """
    Rank candidate documents by field-weighted BM25 relevance score.

    Parameters
    ----------
    query_terms : list[str]
        Normalised query tokens (from query parser).
    candidate_doc_ids : list[int]
        Document IDs returned by candidate retrieval.
    index : FieldIndex
        The field-aware inverted index.
    k1 : float
        Term-frequency saturation parameter.
    b : float
        Document-length normalization parameter.
    title_weight : float
        Weight multiplier for title BM25 scores.
    body_weight : float
        Weight multiplier for body BM25 scores.

    Returns
    -------
    list[RankedCandidate]
        Candidates sorted by:
        1. Weighted BM25 score — descending
        2. Document ID — ascending (tie-breaker)
    """
    if not query_terms or not candidate_doc_ids:
        return []

    total_docs = index.get_total_documents()
    if total_docs == 0:
        return []

    # Deduplicate query terms.
    unique_query_terms = set(query_terms)

    # ── Pre-compute per-field IDFs ────────────────────────────
    fields_config = [
        (FIELD_TITLE, title_weight),
        (FIELD_BODY, body_weight),
    ]

    # For each field, pre-compute IDF and average doc length.
    field_data: dict[str, dict] = {}
    for field_name, weight in fields_config:
        if weight == 0:
            continue  # Skip fields with zero weight.

        field_idx = index.get_field_index(field_name)
        avgdl = field_idx.get_average_document_length()

        idfs: dict[str, float] = {}
        for term in unique_query_terms:
            df = field_idx.get_document_frequency(term)
            if df == 0:
                continue
            idfs[term] = bm25_idf(total_docs, df)

        field_data[field_name] = {
            "weight": weight,
            "avgdl": avgdl,
            "idfs": idfs,
            "field_idx": field_idx,
        }

    if not field_data:
        return []

    # Check if any field has any known query terms.
    has_any_idfs = any(fd["idfs"] for fd in field_data.values())
    if not has_any_idfs:
        return []

    # ── Score each candidate ─────────────────────────────────
    results: list[RankedCandidate] = []

    for doc_id in candidate_doc_ids:
        total_score = 0.0
        matched: set[str] = set()

        for field_name, fd in field_data.items():
            field_idx = fd["field_idx"]
            weight = fd["weight"]
            avgdl = fd["avgdl"]
            idfs = fd["idfs"]

            if not idfs or avgdl <= 0:
                continue

            doc_length = field_idx.get_document_length(doc_id)
            field_score = 0.0

            for term, term_idf in idfs.items():
                posting = field_idx.get_posting(term, doc_id)
                if posting is None:
                    continue

                matched.add(term)
                field_score += bm25_term_score(
                    term_frequency=posting.term_frequency,
                    document_length=doc_length,
                    average_document_length=avgdl,
                    idf=term_idf,
                    k1=k1,
                    b=b,
                )

            total_score += weight * field_score

        results.append(
            RankedCandidate(
                document_id=doc_id,
                score=total_score,
                matched_terms=matched,
            )
        )

    # ── Sort: score desc, then doc_id asc ────────────────────
    results.sort(key=lambda r: (-r.score, r.document_id))

    return results
