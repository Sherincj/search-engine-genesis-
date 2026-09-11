"""
Candidate retrieval — find matching documents via the field-aware index.

Phase 8 update: now searches across **title and body** fields.

Uses **OR semantics**: a document matches if it contains **any** of
the query terms in **any** field.  This maximises recall.

No scoring happens here — just identification of candidate document IDs
with match metadata for the ranking stage.
"""

from __future__ import annotations

from app.indexing.field_index import FieldIndex
from app.search.types import CandidateMatch


def retrieve_candidates(
    index: FieldIndex,
    query_terms: list[str],
) -> list[CandidateMatch]:
    """
    Retrieve candidate documents from the field-aware index.

    A document is a candidate if any query term appears in its
    title OR body.

    Parameters
    ----------
    index : FieldIndex
        The field-aware inverted index to search.
    query_terms : list[str]
        Normalised query tokens (from the query parser).
        Duplicates are deduplicated internally.

    Returns
    -------
    list[CandidateMatch]
        One entry per matching document.  Unordered — the ranker
        will sort them.
    """
    # Deduplicate query terms (duplicates should NOT inflate match count).
    unique_terms = set(query_terms)

    # Collect candidate matches by document ID.
    candidates: dict[int, CandidateMatch] = {}

    for term in unique_terms:
        # Search across all fields (title + body).
        for posting in index.get_all_postings(term):
            doc_id = posting.document_id

            if doc_id not in candidates:
                candidates[doc_id] = CandidateMatch(document_id=doc_id)

            candidates[doc_id].matched_terms.add(term)
            candidates[doc_id].total_term_frequency += posting.term_frequency

    return list(candidates.values())
