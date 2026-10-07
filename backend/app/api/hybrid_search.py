"""Hybrid search API — endpoint combining BM25 + semantic retrieval."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.config.settings import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["hybrid-search"])


class HybridSearchResultItem(BaseModel):
    document_id: int
    score: float  # hybrid fused score
    lexical_score: float | None = None
    semantic_score: float | None = None


class HybridSearchResponse(BaseModel):
    query: str
    results: list[HybridSearchResultItem]
    total: int
    took_ms: float
    lexical_ms: float | None = None
    semantic_ms: float | None = None
    fusion_ms: float | None = None


# Lazy-initialized service
_hybrid_engine = None


def _get_hybrid_engine():
    """Lazy-load the hybrid search engine."""
    global _hybrid_engine
    if _hybrid_engine is not None:
        return _hybrid_engine

    if not settings.EMBEDDING_ENABLED:
        return None

    try:
        from app.embeddings.sentence_transformer import SentenceTransformerProvider
        from app.hybrid_search.fusion import WeightedLinearFusion
        from app.hybrid_search.hybrid_search_engine import HybridSearchEngine
        from app.search.search_engine import SearchEngine
        from app.search.semantic_search_engine import SemanticSearchEngine
        from app.vector_index.vector_storage import VectorStorage
        from app.services.search_service import get_search_service

        # Get existing lexical engine from the search service.
        search_service = get_search_service()
        lexical_engine = search_service._engine

        # Load vector index.
        storage = VectorStorage(settings.VECTOR_INDEX_PATH)
        if not storage.exists():
            logger.warning("Vector index not found — hybrid search unavailable")
            return None

        provider = SentenceTransformerProvider(
            model_name=settings.EMBEDDING_MODEL,
            normalize=settings.EMBEDDING_NORMALIZE,
        )

        index, metadata = storage.load(
            expected_model=settings.EMBEDDING_MODEL,
        )

        semantic_engine = SemanticSearchEngine(index, provider)

        fusion = WeightedLinearFusion(
            lexical_weight=settings.HYBRID_LEXICAL_WEIGHT,
            semantic_weight=settings.HYBRID_SEMANTIC_WEIGHT,
        )

        _hybrid_engine = HybridSearchEngine(
            lexical_engine,
            semantic_engine,
            fusion=fusion,
            candidate_limit=settings.HYBRID_CANDIDATE_LIMIT,
        )

        logger.info(
            "Hybrid search ready: weights=%.2f/%.2f, candidates=%d",
            settings.HYBRID_LEXICAL_WEIGHT,
            settings.HYBRID_SEMANTIC_WEIGHT,
            settings.HYBRID_CANDIDATE_LIMIT,
        )
        return _hybrid_engine

    except Exception as exc:
        logger.error("Failed to initialize hybrid search: %s", exc)
        return None


@router.get("/hybrid")
async def hybrid_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    page: int = Query(1, ge=1, description="Page number"),
    debug: bool = Query(False, description="Include debug scores"),
) -> HybridSearchResponse:
    """Execute a hybrid (BM25 + semantic) search."""
    engine = _get_hybrid_engine()

    if engine is None:
        return HybridSearchResponse(
            query=q, results=[], total=0, took_ms=0.0,
        )

    offset = (page - 1) * limit
    result = engine.search(q, limit=limit, offset=offset)

    items = []
    for r in result.ranked:
        item = HybridSearchResultItem(
            document_id=r.document_id,
            score=round(r.hybrid_score, 6),
        )
        if debug:
            item.lexical_score = round(r.lexical_score, 6)
            item.semantic_score = round(r.semantic_score, 6)
        items.append(item)

    return HybridSearchResponse(
        query=q,
        results=items,
        total=result.total,
        took_ms=result.took_ms,
        lexical_ms=result.lexical_ms,
        semantic_ms=result.semantic_ms,
        fusion_ms=result.fusion_ms,
    )
