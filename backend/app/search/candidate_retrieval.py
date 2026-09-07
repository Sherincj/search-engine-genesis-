"""
Candidate retrieval — find matching documents via the inverted index.

Uses **OR semantics**: a document matches if it contains **any** of
the query terms.  This maximises recall for the baseline search.

No scoring happens here — just identification of candidate document IDs
with match metadata for the ranking stage.
"""

from __future__ import annotations

from app.indexing.inverted_index import InvertedIndex
from app.search.types import CandidateMatch


def retrieve_candidates(
    index: InvertedIndex,
    query_terms: list[str],
) -> list[CandidateMatch]:
    """
    Retrieve candidate documents from the inverted index.

    Parameters
    ----------
    index : InvertedIndex
        The inverted index to search.
    query_terms : list[str]
        Normalised query tokens (from the query parser).
        Duplicates are deduplicated internally.

    Returns
    -------
    list[CandidateMatch]
        One entry per matching document.  Unordered — the ranker
        will sort them.

    Examples
    --------
    >>> retrieve_candidates(index, ["python", "web"])
    [CandidateMatch(document_id=1, ...), CandidateMatch(document_id=2, ...)]
    """
    # Deduplicate query terms (duplicates should NOT inflate match count).
    unique_terms = set(query_terms)

    # Collect candidate matches by document ID.
    candidates: dict[int, CandidateMatch] = {}

    for term in unique_terms:
        for posting in index.get_postings(term):
            doc_id = posting.document_id

            if doc_id not in candidates:
                candidates[doc_id] = CandidateMatch(document_id=doc_id)

            candidates[doc_id].matched_terms.add(term)
            candidates[doc_id].total_term_frequency += posting.term_frequency

    return list(candidates.values())
