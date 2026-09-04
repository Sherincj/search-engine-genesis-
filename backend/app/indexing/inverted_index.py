"""
Inverted index — core data structure.

An inverted index reverses the document→term relationship so that
term lookups are O(1) instead of requiring a full document scan::

    Without index:  "python" → scan every document
    With index:     "python" → [doc 1, doc 4, doc 17]   (direct lookup)

Each term maps to a **posting list** — an ordered sequence of
``Posting`` objects, each recording which document contains the term,
how often (term frequency), and at which positions.

This structure stores everything future ranking algorithms need:

* TF  → ``Posting.term_frequency``
* DF  → ``len(postings)`` for a term
* N   → ``total_documents``
* |D| → ``document_length``
* avgdl → ``average_document_length``
* positions → phrase / proximity search (future)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Posting ──────────────────────────────────────────────────

@dataclass
class Posting:
    """
    A single entry in a term's posting list.

    Records that *document_id* contains the term *term_frequency*
    times, at the given *positions* (zero-indexed token offsets).

    Example::

        Posting(document_id=1, term_frequency=2, positions=[0, 5])
        # "python" appears twice in document 1, at positions 0 and 5.
    """

    document_id: int
    term_frequency: int
    positions: list[int] = field(default_factory=list)


# ── InvertedIndex ────────────────────────────────────────────

class InvertedIndex:
    """
    In-memory inverted index mapping terms to posting lists.

    The index also maintains document statistics (lengths, counts)
    required by TF-IDF and BM25 in later phases.

    Usage::

        idx = InvertedIndex()
        idx.add_document(1, ["python", "web", "python"])
        idx.get_postings("python")
        # [Posting(document_id=1, term_frequency=2, positions=[0, 2])]
    """

    def __init__(self) -> None:
        # term → {doc_id → Posting}
        self._postings: dict[str, dict[int, Posting]] = {}
        # doc_id → token count
        self._doc_lengths: dict[int, int] = {}
        # doc_id → set of terms present (speeds up removal)
        self._doc_terms: dict[int, set[str]] = {}

    # ── Add / update ─────────────────────────────────────────

    def add_document(self, document_id: int, tokens: list[str]) -> None:
        """
        Index a document's tokens.

        If the document already exists it is **replaced** — old postings
        are removed first, preventing duplicates.

        Parameters
        ----------
        document_id : int
            Unique document identifier (typically the DB primary key).
        tokens : list[str]
            Pre-normalised token sequence from the tokeniser.
        """
        # Replace existing document (idempotent re-indexing).
        if document_id in self._doc_lengths:
            self.remove_document(document_id)

        # Build per-term positions map.
        term_positions: dict[str, list[int]] = {}
        for position, token in enumerate(tokens):
            term_positions.setdefault(token, []).append(position)

        # Insert postings.
        terms_in_doc: set[str] = set()
        for term, positions in term_positions.items():
            bucket = self._postings.setdefault(term, {})
            bucket[document_id] = Posting(
                document_id=document_id,
                term_frequency=len(positions),
                positions=positions,
            )
            terms_in_doc.add(term)

        # Record document metadata.
        self._doc_lengths[document_id] = len(tokens)
        self._doc_terms[document_id] = terms_in_doc

    # ── Remove ───────────────────────────────────────────────

    def remove_document(self, document_id: int) -> bool:
        """
        Remove a document from the index.

        Returns ``True`` if the document was present, ``False`` otherwise.
        Empty posting lists are cleaned up automatically.
        """
        if document_id not in self._doc_lengths:
            return False

        for term in self._doc_terms[document_id]:
            bucket = self._postings.get(term)
            if bucket is not None:
                bucket.pop(document_id, None)
                if not bucket:
                    del self._postings[term]

        del self._doc_lengths[document_id]
        del self._doc_terms[document_id]
        return True

    # ── Lookups ──────────────────────────────────────────────

    def get_postings(self, term: str) -> list[Posting]:
        """
        Return the posting list for *term*, sorted by document ID.

        Returns an empty list for unknown terms (no exception).
        """
        bucket = self._postings.get(term)
        if not bucket:
            return []
        return sorted(bucket.values(), key=lambda p: p.document_id)

    def get_posting(self, term: str, document_id: int) -> Posting | None:
        """Direct O(1) lookup of a single posting."""
        bucket = self._postings.get(term)
        if bucket is None:
            return None
        return bucket.get(document_id)

    def contains_term(self, term: str) -> bool:
        """Return ``True`` if at least one document contains *term*."""
        return term in self._postings and len(self._postings[term]) > 0

    # ── Statistics ───────────────────────────────────────────

    def get_document_frequency(self, term: str) -> int:
        """Number of distinct documents containing *term*."""
        bucket = self._postings.get(term)
        return len(bucket) if bucket else 0

    def get_document_length(self, document_id: int) -> int:
        """Token count for a specific document (0 if unknown)."""
        return self._doc_lengths.get(document_id, 0)

    def get_total_documents(self) -> int:
        """Total number of indexed documents."""
        return len(self._doc_lengths)

    def get_average_document_length(self) -> float:
        """Mean token count across all indexed documents."""
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    @property
    def vocabulary(self) -> set[str]:
        """Set of all indexed terms."""
        return set(self._postings.keys())

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct indexed terms."""
        return len(self._postings)

    # ── Housekeeping ─────────────────────────────────────────

    def clear(self) -> None:
        """Remove all data from the index."""
        self._postings.clear()
        self._doc_lengths.clear()
        self._doc_terms.clear()

    def has_document(self, document_id: int) -> bool:
        """Return ``True`` if *document_id* is currently indexed."""
        return document_id in self._doc_lengths

    # ── Validation ───────────────────────────────────────────

    def validate(self) -> list[str]:
        """
        Check internal consistency.

        Returns a list of error descriptions.  An empty list means
        the index is consistent.  Useful during development.
        """
        errors: list[str] = []

        for term, bucket in self._postings.items():
            if not bucket:
                errors.append(
                    f"Term '{term}': empty posting list should be removed"
                )
            for doc_id, posting in bucket.items():
                if posting.document_id != doc_id:
                    errors.append(
                        f"Term '{term}', doc {doc_id}: "
                        f"posting.document_id={posting.document_id} mismatch"
                    )
                if posting.term_frequency != len(posting.positions):
                    errors.append(
                        f"Term '{term}', doc {doc_id}: "
                        f"TF={posting.term_frequency} but "
                        f"{len(posting.positions)} positions"
                    )
                if posting.positions != sorted(posting.positions):
                    errors.append(
                        f"Term '{term}', doc {doc_id}: positions not sorted"
                    )
                if doc_id not in self._doc_lengths:
                    errors.append(
                        f"Term '{term}', doc {doc_id}: "
                        f"no document length recorded"
                    )

        for doc_id, terms in self._doc_terms.items():
            for term in terms:
                if term not in self._postings:
                    errors.append(
                        f"Doc {doc_id}: term '{term}' in doc_terms "
                        f"but not in postings"
                    )
                elif doc_id not in self._postings[term]:
                    errors.append(
                        f"Doc {doc_id}: term '{term}' in doc_terms "
                        f"but posting missing"
                    )

        return errors

    # ── Serialisation helpers (used by IndexStorage) ─────────

    def to_dict(self) -> dict:
        """Export the full index state as a JSON-serialisable dict."""
        return {
            "postings": {
                term: [
                    {
                        "document_id": p.document_id,
                        "term_frequency": p.term_frequency,
                        "positions": p.positions,
                    }
                    for p in sorted(bucket.values(), key=lambda p: p.document_id)
                ]
                for term, bucket in self._postings.items()
            },
            "doc_lengths": {
                str(k): v for k, v in self._doc_lengths.items()
            },
            "doc_terms": {
                str(k): sorted(v) for k, v in self._doc_terms.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> InvertedIndex:
        """Reconstruct an index from a dict produced by ``to_dict``."""
        idx = cls()

        for term, posting_list in data.get("postings", {}).items():
            bucket: dict[int, Posting] = {}
            for entry in posting_list:
                doc_id = int(entry["document_id"])
                bucket[doc_id] = Posting(
                    document_id=doc_id,
                    term_frequency=int(entry["term_frequency"]),
                    positions=list(entry["positions"]),
                )
            idx._postings[term] = bucket

        idx._doc_lengths = {
            int(k): int(v)
            for k, v in data.get("doc_lengths", {}).items()
        }

        idx._doc_terms = {
            int(k): set(v)
            for k, v in data.get("doc_terms", {}).items()
        }

        return idx
