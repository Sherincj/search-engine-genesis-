"""
Vector index persistence — save/load vector index to disk.

Storage format:
    data/vector_index/
    ├── vectors.npz     — numpy arrays: ids + vectors
    └── metadata.json   — model name, dimension, version, etc.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.vector_index.types import VectorIndexMetadata
from app.vector_index.vector_index import VectorIndex, VectorIndexError
from app.utils.logging import get_logger

logger = get_logger(__name__)

VECTOR_INDEX_VERSION = 1


class VectorStorageError(Exception):
    """Raised for vector storage errors."""


class VectorStorage:
    """
    Persist and load a VectorIndex to/from disk.

    Parameters
    ----------
    path : str | Path
        Directory to store vector index files.
    """

    def __init__(self, path: str | Path, *, version: int = VECTOR_INDEX_VERSION) -> None:
        self._path = Path(path)
        if version <= 0:
            raise VectorStorageError("Vector index version must be positive")
        self._version = version

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        """Check if a persisted vector index exists."""
        return (self._path / "metadata.json").exists()

    def save(
        self,
        index: VectorIndex,
        *,
        model_name: str,
        normalize: bool,
    ) -> None:
        """
        Save a vector index to disk.

        Creates the directory if it doesn't exist.
        """
        self._path.mkdir(parents=True, exist_ok=True)

        doc_ids = index.get_all_ids()
        if doc_ids:
            ids_array = np.array(doc_ids, dtype=np.int64)
            vectors_list = [index.get_vector(did) for did in doc_ids]
            vectors_array = np.array(vectors_list, dtype=np.float32)
        else:
            ids_array = np.array([], dtype=np.int64)
            vectors_array = np.zeros((0, index.dimension), dtype=np.float32)

        np.savez(
            self._path / "vectors.npz",
            ids=ids_array,
            vectors=vectors_array,
        )

        metadata = {
            "version": self._version,
            "model_name": model_name,
            "dimension": index.dimension,
            "normalize": normalize,
            "document_count": index.count(),
        }
        with open(self._path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Vector index saved: %d documents, dimension %d",
            index.count(),
            index.dimension,
        )

    def load(
        self,
        *,
        expected_model: str | None = None,
        expected_dimension: int | None = None,
        expected_normalize: bool | None = None,
        expected_version: int | None = None,
    ) -> tuple[VectorIndex, VectorIndexMetadata]:
        """
        Load a vector index from disk.

        Parameters
        ----------
        expected_model : str | None
            If set, verify the stored model matches.
        expected_dimension : int | None
            If set, verify the stored dimension matches.

        Returns
        -------
        tuple[VectorIndex, VectorIndexMetadata]
            The loaded index and its metadata.

        Raises
        ------
        VectorStorageError
            If the index is missing, corrupt, or incompatible.
        """
        meta_path = self._path / "metadata.json"
        vectors_path = self._path / "vectors.npz"

        if not meta_path.exists():
            raise VectorStorageError(f"Metadata file not found: {meta_path}")
        if not vectors_path.exists():
            raise VectorStorageError(f"Vectors file not found: {vectors_path}")

        # Load metadata.
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            raise VectorStorageError(f"Corrupt metadata: {e}") from e

        expected_version = self._version if expected_version is None else expected_version
        # Version check.
        stored_version = meta.get("version", 0)
        if stored_version != expected_version:
            raise VectorStorageError(
                f"Incompatible vector index version: "
                f"stored={stored_version}, expected={expected_version}"
            )

        stored_model = meta.get("model_name", "")
        stored_dim = meta.get("dimension", 0)
        stored_norm = meta.get("normalize", True)
        stored_count = meta.get("document_count", 0)

        # Model compatibility check.
        if expected_model and stored_model != expected_model:
            raise VectorStorageError(
                f"Incompatible embedding model: "
                f"stored='{stored_model}', expected='{expected_model}'"
            )

        # Dimension compatibility check.
        if expected_dimension and stored_dim != expected_dimension:
            raise VectorStorageError(
                f"Incompatible vector dimension: "
                f"stored={stored_dim}, expected={expected_dimension}"
            )
        if expected_normalize is not None and stored_norm != expected_normalize:
            raise VectorStorageError(
                f"Incompatible embedding normalization: "
                f"stored={stored_norm}, expected={expected_normalize}"
            )

        if not isinstance(stored_dim, int) or stored_dim <= 0:
            raise VectorStorageError(f"Invalid stored vector dimension: {stored_dim!r}")
        if not isinstance(stored_model, str) or not stored_model:
            raise VectorStorageError("Invalid stored embedding model metadata")
        if not isinstance(stored_norm, bool):
            raise VectorStorageError("Invalid stored normalization metadata")
        if not isinstance(stored_count, int) or stored_count < 0:
            raise VectorStorageError("Invalid stored document count metadata")

        # Load vectors.
        try:
            with np.load(vectors_path, allow_pickle=False) as data:
                ids_array = data["ids"]
                vectors_array = data["vectors"]
            if ids_array.ndim != 1:
                raise ValueError("ids must be a one-dimensional array")
            if vectors_array.ndim != 2:
                raise ValueError("vectors must be a two-dimensional array")
            if len(ids_array) != len(vectors_array):
                raise ValueError("ids and vectors have different lengths")
            if vectors_array.shape[1] != stored_dim:
                raise ValueError("stored vector dimension does not match metadata")
            if len(ids_array) != stored_count:
                raise ValueError("stored document count does not match vector data")
        except Exception as e:
            raise VectorStorageError(f"Corrupt vectors file: {e}") from e

        # Build index.
        try:
            index = VectorIndex(dimension=stored_dim)
            for i, doc_id in enumerate(ids_array):
                index.add(int(doc_id), vectors_array[i].tolist())
            index.validate()
        except (TypeError, ValueError, VectorIndexError) as exc:
            raise VectorStorageError(f"Corrupt vector data: {exc}") from exc

        metadata = VectorIndexMetadata(
            model_name=stored_model,
            dimension=stored_dim,
            normalize=stored_norm,
            version=stored_version,
            document_count=index.count(),
        )

        logger.info(
            "Vector index loaded: %d documents, dimension %d, model '%s'",
            index.count(),
            stored_dim,
            stored_model,
        )

        return index, metadata
