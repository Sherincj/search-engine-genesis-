"""
Semantic search engine — vector-based document retrieval.

Uses EmbeddingProvider to convert queries into vectors, then retrieves
similar documents from the VectorIndex using cosine similarity.

This is a SEPARATE search engine from the BM25 lexical engine.
Do NOT use BM25, phrase matching, or query term weighting here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.embeddings.provider import EmbeddingProvider
from app.vector_index.vector_index import VectorIndex
from app.vector_index.types import VectorSearchResult


@dataclass
class SemanticSearchResult:
    """Result from semantic search."""
    query: str
    ranked: list[VectorSearchResult]
    total: int
    took_ms: float
    embedding_ms: float
    search_ms: float


class SemanticSearchEngine:
    """
    Vector-based semantic search engine.

    Parameters
    ----------
    index : VectorIndex
        The vector index to search.
    provider : EmbeddingProvider
        The embedding provider for query encoding.
    """

    def __init__(
        self,
        index: VectorIndex,
        provider: EmbeddingProvider,
    ) -> None:
        if index.dimension != provider.dimension:
            raise ValueError(
                "Vector index dimension does not match embedding provider: "
                f"{index.dimension} != {provider.dimension}"
            )
        self._index = index
        self._provider = provider

    @property
    def index(self) -> VectorIndex:
        return self._index

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> SemanticSearchResult:
        """
        Execute a semantic search.

        Parameters
        ----------
        query : str
            The search query text.
        limit : int
            Maximum number of results.
        offset : int
            Number of results to skip (for pagination).

        Returns
        -------
        SemanticSearchResult
            Ranked results with cosine similarity scores.
        """
        start = time.perf_counter()

        query = query.strip()
        if not query:
            return SemanticSearchResult(
                query=query,
                ranked=[],
                total=0,
                took_ms=0.0,
                embedding_ms=0.0,
                search_ms=0.0,
            )

        # Embed the query.
        embed_start = time.perf_counter()
        query_vector = self._provider.embed_text(query)
        embed_ms = (time.perf_counter() - embed_start) * 1000

        # Search the vector index.
        search_start = time.perf_counter()
        # Retrieve more than needed for pagination. ``total`` still describes
        # the complete index rather than this limited result slice.
        all_results = self._index.search(query_vector, top_k=offset + limit)
        search_ms = (time.perf_counter() - search_start) * 1000

        # Apply pagination.
        paginated = all_results[offset:offset + limit]

        total_ms = (time.perf_counter() - start) * 1000

        return SemanticSearchResult(
            query=query,
            ranked=paginated,
            total=self._index.count(),
            took_ms=round(total_ms, 2),
            embedding_ms=round(embed_ms, 2),
            search_ms=round(search_ms, 2),
        )
