"""
Vector index types — shared types for the vector search system.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VectorSearchResult:
    """A single result from vector similarity search."""
    document_id: int
    score: float  # cosine similarity


@dataclass(frozen=True)
class VectorIndexMetadata:
    """Metadata about a vector index."""
    model_name: str
    dimension: int
    normalize: bool
    version: int
    document_count: int
