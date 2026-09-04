"""
Index persistence — save and load the inverted index to/from disk.

Uses JSON serialisation for V1.  Human-readable, debuggable, and
sufficient for 1K–10K documents.  The ``IndexStorage`` abstraction
allows the format to be replaced later (e.g. with a binary format
or a distributed index store).

Atomic writes
-------------
To avoid corruption on interrupted saves, the index is written to a
temporary file first, then atomically renamed over the target::

    write to index.json.tmp  →  flush  →  rename to index.json
"""

from __future__ import annotations

import json
from pathlib import Path

from app.indexing.inverted_index import InvertedIndex
from app.utils.logging import get_logger

logger = get_logger(__name__)

_INDEX_FILENAME = "index.json"
_TEMP_SUFFIX = ".tmp"


class IndexStorageError(Exception):
    """Raised when the index cannot be loaded (corrupt, missing, etc.)."""

    pass


class IndexStorage:
    """
    Persists an ``InvertedIndex`` to the local filesystem.

    Parameters
    ----------
    index_path : str | Path
        Directory where the index file is stored.

    Usage::

        storage = IndexStorage("data/index")
        storage.save(index)
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

    def save(self, index: InvertedIndex) -> None:
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
            "Index saved to %s (%d terms, %d documents)",
            self._index_file,
            index.vocabulary_size,
            index.get_total_documents(),
        )

    def load(self) -> InvertedIndex:
        """
        Load an index from disk.

        Raises
        ------
        IndexStorageError
            If the file is missing or corrupt.
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

        index = InvertedIndex.from_dict(data)
        logger.info(
            "Index loaded from %s (%d terms, %d documents)",
            self._index_file,
            index.vocabulary_size,
            index.get_total_documents(),
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
