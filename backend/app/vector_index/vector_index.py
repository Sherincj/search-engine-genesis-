"""
Custom vector index — exact cosine similarity search.

Implements O(N×D) brute-force search for correctness and understanding.
No approximate nearest-neighbor algorithms (HNSW/FAISS) are used.

Supports:
- add / update / remove documents
- cosine similarity retrieval
- deterministic tie-breaking (score DESC, document_id ASC)
- dimension and value validation
"""

from __future__ import annotations

import math
from typing import Sequence

from app.vector_index.types import VectorSearchResult


class VectorIndexError(Exception):
    """Raised for vector index validation errors."""


class VectorIndex:
    """
    In-memory exact vector search index.

    Parameters
    ----------
    dimension : int
        Expected vector dimension. All vectors must match this.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise VectorIndexError(f"Dimension must be positive, got {dimension}")
        self._dimension = dimension
        # doc_id -> vector (as list[float])
        self._vectors: dict[int, list[float]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, document_id: int, vector: list[float]) -> None:
        """
        Add or replace a document vector.

        If the document_id already exists, the vector is replaced.
        """
        self._validate_document_id(document_id)
        self._validate_vector(vector)
        self._vectors[document_id] = list(vector)  # defensive copy

    def remove(self, document_id: int) -> bool:
        """
        Remove a document vector.

        Returns True if the document was removed, False if not found.
        """
        if document_id in self._vectors:
            del self._vectors[document_id]
            return True
        return False

    def update(self, document_id: int, vector: list[float]) -> None:
        """Update (replace) a document vector. Same as add()."""
        self.add(document_id, vector)

    def contains(self, document_id: int) -> bool:
        """Check if a document is in the index."""
        return document_id in self._vectors

    def count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._vectors)

    def clear(self) -> None:
        """Remove all vectors from the index."""
        self._vectors.clear()

    def get_vector(self, document_id: int) -> list[float] | None:
        """Return a copy of the vector for a document, or None."""
        vec = self._vectors.get(document_id)
        if vec is not None:
            return list(vec)
        return None

    def get_all_ids(self) -> list[int]:
        """Return all indexed document IDs."""
        return list(self._vectors.keys())

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        """
        Find the top-K most similar documents by cosine similarity.

        Parameters
        ----------
        query_vector : list[float]
            The query embedding vector.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[VectorSearchResult]
            Results sorted by similarity DESC, document_id ASC.
        """
        self._validate_vector(query_vector)
        if top_k <= 0:
            return []

        results: list[VectorSearchResult] = []
        for doc_id, doc_vector in self._vectors.items():
            sim = cosine_similarity(query_vector, doc_vector)
            results.append(VectorSearchResult(document_id=doc_id, score=sim))

        # Sort: score DESC, document_id ASC for determinism.
        results.sort(key=lambda r: (-r.score, r.document_id))
        return results[:top_k]

    def validate(self) -> None:
        """
        Validate internal consistency.

        Raises VectorIndexError if any issues are found.
        """
        for doc_id, vector in self._vectors.items():
            if len(vector) != self._dimension:
                raise VectorIndexError(
                    f"Document {doc_id}: vector dimension {len(vector)} "
                    f"!= index dimension {self._dimension}"
                )
            for i, val in enumerate(vector):
                if not math.isfinite(val):
                    raise VectorIndexError(
                        f"Document {doc_id}: non-finite value at index {i}: {val}"
                    )

    def _validate_document_id(self, document_id: int) -> None:
        if not isinstance(document_id, int) or document_id < 0:
            raise VectorIndexError(
                f"Document ID must be a non-negative integer, got {document_id}"
            )

    def _validate_vector(self, vector: list[float] | Sequence[float]) -> None:
        if not vector:
            raise VectorIndexError("Vector must not be empty")
        if len(vector) != self._dimension:
            raise VectorIndexError(
                f"Vector dimension {len(vector)} does not match "
                f"index dimension {self._dimension}"
            )
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                raise VectorIndexError(
                    f"Vector value at index {i} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise VectorIndexError(
                    f"Vector contains non-finite value at index {i}: {val}"
                )


def cosine_similarity(
    vector_a: list[float] | Sequence[float],
    vector_b: list[float] | Sequence[float],
) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine(A, B) = (A · B) / (|A| × |B|)

    Returns 0.0 if either vector has zero magnitude.

    Parameters
    ----------
    vector_a, vector_b : list[float]
        Vectors of equal length.

    Returns
    -------
    float
        Cosine similarity in range [-1, 1].
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vector dimensions must match: {len(vector_a)} != {len(vector_b)}"
        )

    dot_product = 0.0
    magnitude_a = 0.0
    magnitude_b = 0.0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b
        magnitude_a += a * a
        magnitude_b += b * b

    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
