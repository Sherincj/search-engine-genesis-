"""
Field-aware inverted index — wraps per-field ``InvertedIndex`` instances.

Phase 8 introduces field-level search.  Instead of modifying the
battle-tested ``InvertedIndex``, this module wraps two instances::

    FieldIndex
    ├── title_index  (InvertedIndex)
    └── body_index   (InvertedIndex)

Each field maintains independent statistics (TF, DF, doc lengths,
average doc length) which is critical for correct BM25 field scoring.

The ``FieldIndex`` also provides a unified candidate-retrieval view
that merges document IDs across both fields.
"""

from __future__ import annotations

from app.indexing.inverted_index import InvertedIndex, Posting


# Supported field names — kept as module-level constants for reuse.
FIELD_TITLE = "title"
FIELD_BODY = "body"
_ALL_FIELDS = frozenset({FIELD_TITLE, FIELD_BODY})

# Current serialisation version.
INDEX_VERSION = 2


class FieldIndex:
    """
    Field-aware index wrapping per-field ``InvertedIndex`` instances.

    Parameters
    ----------
    title_index : InvertedIndex | None
        Pre-built title index.  A fresh empty index is created if omitted.
    body_index : InvertedIndex | None
        Pre-built body index.  A fresh empty index is created if omitted.

    Usage::

        fi = FieldIndex()
        fi.add_document(1, title_tokens=[\"python\"], body_tokens=[\"web\", \"dev\"])
    """

    def __init__(
        self,
        title_index: InvertedIndex | None = None,
        body_index: InvertedIndex | None = None,
    ) -> None:
        self._indexes: dict[str, InvertedIndex] = {
            FIELD_TITLE: title_index or InvertedIndex(),
            FIELD_BODY: body_index or InvertedIndex(),
        }

    # ── Convenience properties ───────────────────────────────

    @property
    def title_index(self) -> InvertedIndex:
        return self._indexes[FIELD_TITLE]

    @property
    def body_index(self) -> InvertedIndex:
        return self._indexes[FIELD_BODY]

    def get_field_index(self, field: str) -> InvertedIndex:
        """Return the ``InvertedIndex`` for *field*.

        Raises ``ValueError`` for unknown fields.
        """
        if field not in _ALL_FIELDS:
            raise ValueError(
                f"Unknown field '{field}'. Must be one of {sorted(_ALL_FIELDS)}"
            )
        return self._indexes[field]

    # ── Document operations ──────────────────────────────────

    def add_document(
        self,
        document_id: int,
        *,
        title_tokens: list[str] | None = None,
        body_tokens: list[str] | None = None,
    ) -> None:
        """
        Index a document's title and body tokens.

        Empty or None token lists are safe — the field simply gets
        zero-length entries.
        """
        self._indexes[FIELD_TITLE].add_document(
            document_id, title_tokens or []
        )
        self._indexes[FIELD_BODY].add_document(
            document_id, body_tokens or []
        )

    def remove_document(self, document_id: int) -> bool:
        """Remove a document from all field indexes."""
        title_removed = self._indexes[FIELD_TITLE].remove_document(document_id)
        body_removed = self._indexes[FIELD_BODY].remove_document(document_id)
        return title_removed or body_removed

    def has_document(self, document_id: int) -> bool:
        """Return True if the document exists in any field index."""
        return any(idx.has_document(document_id) for idx in self._indexes.values())

    # ── Field-aware lookups ──────────────────────────────────

    def get_postings(self, term: str, field: str) -> list[Posting]:
        """Return postings for *term* in *field*."""
        return self.get_field_index(field).get_postings(term)

    def get_posting(self, term: str, document_id: int, field: str) -> Posting | None:
        """Direct O(1) lookup of a single posting in *field*."""
        return self.get_field_index(field).get_posting(term, document_id)

    # ── Field-aware statistics ───────────────────────────────

    def get_document_frequency(self, term: str, field: str) -> int:
        """DF for *term* in *field*."""
        return self.get_field_index(field).get_document_frequency(term)

    def get_document_length(self, document_id: int, field: str) -> int:
        """Token count for *document_id* in *field*."""
        return self.get_field_index(field).get_document_length(document_id)

    def get_average_document_length(self, field: str) -> float:
        """Mean token count across all documents for *field*."""
        return self.get_field_index(field).get_average_document_length()

    def get_total_documents(self) -> int:
        """Total indexed documents (same across both fields)."""
        # Both fields are populated for every document, so they agree.
        return self._indexes[FIELD_BODY].get_total_documents()

    # ── Cross-field candidate retrieval helper ────────────────

    def get_all_postings(self, term: str) -> list[Posting]:
        """
        Return postings for *term* across **all** fields (union).

        Used by candidate retrieval: a document is a candidate if
        the term appears in title OR body.  Duplicate document IDs
        are deduplicated (title posting preferred for the match object,
        but callers only need the doc IDs).
        """
        seen: dict[int, Posting] = {}
        for idx in self._indexes.values():
            for posting in idx.get_postings(term):
                if posting.document_id not in seen:
                    seen[posting.document_id] = posting
        return sorted(seen.values(), key=lambda p: p.document_id)

    # ── Vocabulary ────────────────────────────────────────────

    @property
    def vocabulary(self) -> set[str]:
        """Union of all terms across all fields."""
        result: set[str] = set()
        for idx in self._indexes.values():
            result |= idx.vocabulary
        return result

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    # ── Housekeeping ──────────────────────────────────────────

    def clear(self) -> None:
        for idx in self._indexes.values():
            idx.clear()

    # ── Validation ────────────────────────────────────────────

    def validate(self) -> list[str]:
        """
        Validate all field indexes and cross-field consistency.

        Returns a list of error descriptions (empty = valid).
        """
        errors: list[str] = []

        for field_name, idx in self._indexes.items():
            for err in idx.validate():
                errors.append(f"[{field_name}] {err}")

        # Check that both fields index the same set of document IDs.
        title_docs = set(self._indexes[FIELD_TITLE]._doc_lengths.keys())
        body_docs = set(self._indexes[FIELD_BODY]._doc_lengths.keys())

        title_only = title_docs - body_docs
        body_only = body_docs - title_docs

        for doc_id in title_only:
            errors.append(
                f"Doc {doc_id}: present in title index but missing from body index"
            )
        for doc_id in body_only:
            errors.append(
                f"Doc {doc_id}: present in body index but missing from title index"
            )

        return errors

    # ── Serialisation ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Export as a JSON-serialisable dict with version marker."""
        return {
            "index_version": INDEX_VERSION,
            "fields": {
                field_name: idx.to_dict()
                for field_name, idx in self._indexes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> FieldIndex:
        """
        Reconstruct from a dict produced by ``to_dict``.

        Raises
        ------
        ValueError
            If the dict is from an older (pre-field-aware) format.
        """
        version = data.get("index_version")
        if version is None or version < INDEX_VERSION:
            raise ValueError(
                f"Index format version is outdated "
                f"(found {version!r}, need {INDEX_VERSION}). "
                f"Rebuild the index for field-aware search."
            )

        fields = data.get("fields", {})
        if FIELD_TITLE not in fields or FIELD_BODY not in fields:
            raise ValueError(
                f"Missing required fields in index data. "
                f"Expected '{FIELD_TITLE}' and '{FIELD_BODY}'."
            )

        title_idx = InvertedIndex.from_dict(fields[FIELD_TITLE])
        body_idx = InvertedIndex.from_dict(fields[FIELD_BODY])

        return cls(title_index=title_idx, body_index=body_idx)
