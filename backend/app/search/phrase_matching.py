"""
Phrase matching — verifies exact term adjacency using positional data.

Phase 11: uses ``Posting.positions`` from the Phase 4 inverted index
to detect whether a sequence of terms occurs consecutively in a
document field.

Algorithm:

    For phrase ["python", "development"]:
    1. Get positions of "python" in field: [2, 8, 15]
    2. Get positions of "development" in field: [3, 10, 20]
    3. Check if python_pos + 1 exists in development positions
       → 2 + 1 = 3 ✓  → phrase match at position 2

For multi-term phrases, progressively verify each term.

This module is **independent** of BM25, FastAPI, SQLAlchemy, and
the HTTP layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.indexing.field_index import FIELD_BODY, FIELD_TITLE, FieldIndex
from app.indexing.inverted_index import Posting


@dataclass(frozen=True)
class PhraseMatch:
    """A verified phrase match in a document field."""

    document_id: int
    field: str
    positions: tuple[int, ...]  # starting positions of the phrase


def find_phrase_positions(
    position_lists: list[list[int]],
) -> list[int]:
    """
    Find starting positions where terms appear consecutively.

    Parameters
    ----------
    position_lists : list[list[int]]
        Position list for each term in the phrase, in order.
        Each list must be sorted ascending.

    Returns
    -------
    list[int]
        Starting positions where the phrase occurs.
        Empty if no match.

    Examples
    --------
    >>> find_phrase_positions([[2, 8, 15], [3, 10, 20]])
    [2]  # "python" at 2, "development" at 3
    """
    if not position_lists or any(len(p) == 0 for p in position_lists):
        return []

    if len(position_lists) == 1:
        return list(position_lists[0])

    # Start with the first term's positions.
    candidates = position_lists[0]

    for offset, next_positions in enumerate(position_lists[1:], start=1):
        next_set = set(next_positions)
        candidates = [p for p in candidates if (p + offset) in next_set]
        if not candidates:
            return []

    return candidates


def check_phrase_in_field(
    phrase_terms: list[str] | tuple[str, ...],
    document_id: int,
    field_index,
) -> list[int]:
    """
    Check if a phrase occurs in a specific field for a document.

    Parameters
    ----------
    phrase_terms : list[str] | tuple[str, ...]
        Normalised phrase terms in order.
    document_id : int
        The document to check.
    field_index : InvertedIndex
        The field-specific inverted index.

    Returns
    -------
    list[int]
        Starting positions of the phrase, or empty if no match.
    """
    position_lists: list[list[int]] = []

    for term in phrase_terms:
        posting = field_index.get_posting(term, document_id)
        if posting is None:
            return []
        position_lists.append(posting.positions)

    return find_phrase_positions(position_lists)


def find_phrase_matches(
    phrase_terms: list[str] | tuple[str, ...],
    document_id: int,
    index: FieldIndex,
) -> list[PhraseMatch]:
    """
    Find phrase matches across all fields for a document.

    Parameters
    ----------
    phrase_terms : list[str] | tuple[str, ...]
        Normalised phrase terms in order.
    document_id : int
        The document to check.
    index : FieldIndex
        The field-aware inverted index.

    Returns
    -------
    list[PhraseMatch]
        One entry per field that contains the phrase.
    """
    matches: list[PhraseMatch] = []

    for field_name in (FIELD_TITLE, FIELD_BODY):
        field_idx = index.get_field_index(field_name)
        positions = check_phrase_in_field(phrase_terms, document_id, field_idx)
        if positions:
            matches.append(
                PhraseMatch(
                    document_id=document_id,
                    field=field_name,
                    positions=tuple(positions),
                )
            )

    return matches


def get_phrase_candidates(
    phrase_terms: list[str] | tuple[str, ...],
    index: FieldIndex,
) -> set[int]:
    """
    Get candidate document IDs that might contain a phrase.

    Uses the rarest term's posting list as the starting point,
    then intersects with other terms' document sets.

    Parameters
    ----------
    phrase_terms : list[str] | tuple[str, ...]
        Normalised phrase terms in order.
    index : FieldIndex
        The field-aware inverted index.

    Returns
    -------
    set[int]
        Document IDs that contain all phrase terms in at least one field.
    """
    if not phrase_terms:
        return set()

    # For each field, find docs containing ALL phrase terms.
    result: set[int] = set()

    for field_name in (FIELD_TITLE, FIELD_BODY):
        field_idx = index.get_field_index(field_name)

        # Get doc sets for each term, starting with the rarest.
        term_doc_sets: list[tuple[int, set[int]]] = []
        for term in phrase_terms:
            postings = field_idx.get_postings(term)
            doc_ids = {p.document_id for p in postings}
            term_doc_sets.append((len(doc_ids), doc_ids))

        if not term_doc_sets:
            continue

        # Start with the rarest term (smallest doc set).
        term_doc_sets.sort(key=lambda x: x[0])
        candidates = term_doc_sets[0][1].copy()

        for _, doc_set in term_doc_sets[1:]:
            candidates &= doc_set
            if not candidates:
                break

        result |= candidates

    return result
