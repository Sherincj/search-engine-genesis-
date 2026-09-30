"""
Embedding provider interface — abstract contract for text embedding.

Any embedding implementation (local model, API, fake) must satisfy
this protocol so the vector index and search engine remain decoupled
from the specific embedding library.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.embeddings.types import Embedding


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for text-to-vector embedding providers."""

    def embed_text(self, text: str) -> Embedding:
        """Embed a single text string into a vector."""
        ...

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of text strings into vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Return the vector dimension for this provider."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name for this provider."""
        ...


def build_embedding_text(title: str, body: str) -> str:
    """
    Build the text to embed for a document.

    Concatenates title and body with a newline separator.
    This ensures title semantics are included in the embedding.

    Parameters
    ----------
    title : str
        Document title.
    body : str
        Document body text.

    Returns
    -------
    str
        Combined text for embedding.
    """
    title = title.strip()
    body = body.strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body
