"""
Embedding types — shared type definitions for the embedding system.
"""

from __future__ import annotations

from dataclasses import dataclass

# A vector embedding is a list of floats.
Embedding = list[float]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for an embedding provider."""
    model_name: str
    dimension: int
    normalize: bool = True
