"""
Index persistence — save and load the field-aware index to/from disk.

Phase 8 update: now persists ``FieldIndex`` (version 2) instead of
a bare ``InvertedIndex``.  Old version-1 indexes are detected and
rejected with a clear error message.

Uses JSON serialisation.  Human-readable, debuggable, and sufficient
for 1K–10K documents.

Atomic writes
-------------
To avoid corruption on interrupted saves, the index is written to a
temporary file first, then atomically renamed over the target::

    write to index.json.tmp  →  flush  →  rename to index.json
"""

from __future__ import annotations

import json
from pathlib import Path

from app.indexing.field_index import FieldIndex, INDEX_VERSION
from app.utils.logging import get_logger

logger = get_logger(__name__)

_INDEX_FILENAME = "index.json"
_TEMP_SUFFIX = ".tmp"


class IndexStorageError(Exception):
    """Raised when the index cannot be loaded (corrupt, missing, etc.)."""

    pass


class IndexStorage:
    """
    Persists a ``FieldIndex`` to the local filesystem.

    Parameters
    ----------
    index_path : str | Path
        Directory where the index file is stored.

    Usage::

        storage = IndexStorage("data/index")
        storage.save(field_index)
        loaded = storage.load()
    """

    def __init__(self, index_path: str | Path) -> None:
        self._dir = Path(index_path)

    @property
    def _index_file(self) -> Path:
        return self._dir / _INDEX_FILENAME

    @property
    def _temp_file(self) -> Path:
        return self._dir / f"{_INDEX_FILENAME}{_TEMP_SUFFIX}"

    # ── Public API ───────────────────────────────────────────

    def save(self, index: FieldIndex) -> None:
        """
        Persist *index* to disk.

        Uses atomic write: temp file → flush → rename.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        data = index.to_dict()

        # Write to temp file first.
        with open(self._temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()

        # Atomic rename.
        self._temp_file.replace(self._index_file)
        logger.info(
            "Index saved to %s (%d terms, %d documents, version %d)",
            self._index_file,
            index.vocabulary_size,
            index.get_total_documents(),
            INDEX_VERSION,
        )

    def load(self) -> FieldIndex:
        """
        Load a field-aware index from disk.

        Raises
        ------
        IndexStorageError
            If the file is missing, corrupt, or uses an old format.
        """
        if not self._index_file.exists():
            raise IndexStorageError(
                f"Index file not found: {self._index_file}"
            )

        try:
            with open(self._index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise IndexStorageError(
                f"Corrupt index file: {self._index_file} — {exc}"
            ) from exc

        # ── Version detection ────────────────────────────────
        version = data.get("index_version")
        if version is None:
            # Old Phase 7 (single-field) index detected.
            raise IndexStorageError(
                "Index format is outdated (pre-field-aware). "
                "Rebuild the index for field-aware search."
            )

        try:
            index = FieldIndex.from_dict(data)
        except (ValueError, KeyError, TypeError) as exc:
            raise IndexStorageError(
                f"Invalid index data in {self._index_file}: {exc}"
            ) from exc

        logger.info(
            "Index loaded from %s (%d terms, %d documents, version %d)",
            self._index_file,
            index.vocabulary_size,
            index.get_total_documents(),
            version,
        )
        return index

    def exists(self) -> bool:
        """Return ``True`` if a persisted index exists on disk."""
        return self._index_file.exists()

    def delete(self) -> None:
        """Delete the persisted index file if it exists."""
        if self._index_file.exists():
            self._index_file.unlink()
            logger.info("Index deleted: %s", self._index_file)
        # Clean up temp file too.
        if self._temp_file.exists():
            self._temp_file.unlink()
