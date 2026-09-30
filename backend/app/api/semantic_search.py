"""Separate API endpoint for exact vector-based semantic retrieval."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.database.repositories import DocumentRepository
from app.services.semantic_index_service import (
    SemanticIndexServiceError,
    get_semantic_index_service,
)

router = APIRouter(prefix="/search", tags=["semantic-search"])


class SemanticSearchResultItem(BaseModel):
    document_id: int
    score: float  # Exact cosine similarity; it is not a BM25 score.
    title: str
    url: str


class SemanticSearchResponse(BaseModel):
    query: str
    results: list[SemanticSearchResultItem]
    total: int
    took_ms: float
    embedding_ms: float
    search_ms: float


@router.get("/semantic", response_model=SemanticSearchResponse)
async def semantic_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    page: int = Query(1, ge=1, description="Page number"),
    db: AsyncSession = Depends(get_db),
) -> SemanticSearchResponse:
    """Search the independent semantic/vector index."""
    try:
        service = get_semantic_index_service()
    except SemanticIndexServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = service.engine.search(q, limit=limit, offset=(page - 1) * limit)
    docs = await DocumentRepository(db).get_by_ids(
        [candidate.document_id for candidate in result.ranked]
    )
    docs_by_id = {doc.id: doc for doc in docs}

    # A deleted document can briefly remain in an old persisted index. It is
    # never returned as a search hit, matching lexical-search stale-ID safety.
    items = [
        SemanticSearchResultItem(
            document_id=candidate.document_id,
            score=round(candidate.score, 6),
            title=docs_by_id[candidate.document_id].title,
            url=docs_by_id[candidate.document_id].url,
        )
        for candidate in result.ranked
        if candidate.document_id in docs_by_id
    ]

    return SemanticSearchResponse(
        query=result.query,
        results=items,
        total=result.total,
        took_ms=result.took_ms,
        embedding_ms=result.embedding_ms,
        search_ms=result.search_ms,
    )
