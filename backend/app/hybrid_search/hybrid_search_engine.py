"""
Hybrid search engine — combines BM25 lexical and semantic vector search.

Orchestrates:
1. Parallel candidate retrieval from BM25 and semantic engines
2. Candidate union (merge)
3. Score normalization
4. Weighted linear fusion
5. Final ranking

Does NOT modify the BM25 or semantic engines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.hybrid_search.candidate_merger import merge_candidates
from app.hybrid_search.fusion import FusionStrategy, WeightedLinearFusion
from app.hybrid_search.types import HybridScore
from app.search.search_engine import SearchEngine
from app.search.semantic_search_engine import SemanticSearchEngine
from app.embeddings.provider import EmbeddingProvider
from app.vector_index.vector_index import VectorIndex


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    query: str
    ranked: list[HybridScore]
    total: int
    took_ms: float
    lexical_ms: float
    semantic_ms: float
    fusion_ms: float


class HybridSearchEngine:
    """
    Combines BM25 lexical search with semantic vector search.

    Parameters
    ----------
    lexical_engine : SearchEngine
        The BM25 lexical search engine.
    semantic_engine : SemanticSearchEngine
        The semantic vector search engine.
    fusion : FusionStrategy | None
        Fusion strategy. Defaults to WeightedLinearFusion(0.5, 0.5).
    candidate_limit : int
        Number of candidates to retrieve from each engine.
    """

    def __init__(
        self,
        lexical_engine: SearchEngine,
        semantic_engine: SemanticSearchEngine,
        *,
        fusion: FusionStrategy | None = None,
        candidate_limit: int = 50,
    ) -> None:
        self._lexical = lexical_engine
        self._semantic = semantic_engine
        self._fusion = fusion or WeightedLinearFusion()
        self._candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> HybridSearchResult:
        """
        Execute a hybrid search combining lexical and semantic retrieval.

        Parameters
        ----------
        query : str
            The search query text.
        limit : int
            Maximum number of results to return.
        offset : int
            Number of results to skip (pagination).

        Returns
        -------
        HybridSearchResult
            Fused ranked results.
        """
        start = time.perf_counter()

        query = query.strip()
        if not query:
            return HybridSearchResult(
                query=query, ranked=[], total=0,
                took_ms=0.0, lexical_ms=0.0,
                semantic_ms=0.0, fusion_ms=0.0,
            )

        # 1. Retrieve lexical candidates.
        lex_start = time.perf_counter()
        lex_result = self._lexical.search(query, limit=self._candidate_limit)
        lex_scores = {
            r.document_id: r.score for r in lex_result.ranked
        }
        lexical_ms = (time.perf_counter() - lex_start) * 1000

        # 2. Retrieve semantic candidates.
        sem_start = time.perf_counter()
        sem_result = self._semantic.search(query, limit=self._candidate_limit)
        sem_scores = {
            r.document_id: r.score for r in sem_result.ranked
        }
        semantic_ms = (time.perf_counter() - sem_start) * 1000

        # 3. Merge candidates.
        fusion_start = time.perf_counter()
        candidates = merge_candidates(lex_scores, sem_scores)

        # 4. Fuse scores (normalization + weighting).
        fused = self._fusion.fuse(candidates)

        # 5. Paginate.
        paginated = fused[offset:offset + limit]
        fusion_ms = (time.perf_counter() - fusion_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000

        return HybridSearchResult(
            query=query,
            ranked=paginated,
            total=len(fused),
            took_ms=round(total_ms, 2),
            lexical_ms=round(lexical_ms, 2),
            semantic_ms=round(semantic_ms, 2),
            fusion_ms=round(fusion_ms, 2),
        )
