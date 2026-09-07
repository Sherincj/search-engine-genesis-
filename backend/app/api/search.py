"""
Search API — keyword search endpoint.

Connects the FastAPI router to the ``SearchService`` which
coordinates query parsing, inverted index lookup, ranking,
and database retrieval.
"""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.search import SearchResponse
from app.services.search_service import get_search_service

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(default="", description="Search query"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    limit: int = Query(default=10, ge=1, le=100, description="Results per page"),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse | JSONResponse:
    """
    Search the document index.

    Returns ranked results with pagination.

    - **q** — search query (required, non-empty)
    - **page** — page number, starting from 1
    - **limit** — results per page (1–100)
    """
    # Reject empty/whitespace-only queries.
    if not q or not q.strip():
        return JSONResponse(
            status_code=400,
            content={"detail": "Query parameter 'q' is required."},
        )

    service = get_search_service()
    result = await service.search(q, page=page, limit=limit, db=db)
    return result
