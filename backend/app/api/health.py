"""
Health-check endpoint.

Returns basic system status so operators (and the frontend) can
confirm the backend is alive and which capabilities are ready.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.database.repositories import DocumentRepository
from app.services.search_service import get_search_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Return current system health.

    Reports the real document count from the database and whether
    the database connection is alive.
    """
    try:
        repo = DocumentRepository(db)
        doc_count = await repo.count_documents()
        db_status = "connected"
    except Exception:
        doc_count = 0
        db_status = "error"

    service = get_search_service()
    index_loaded = service.index.get_total_documents() > 0

    return {
        "status": "ok",
        "database": db_status,
        "index_loaded": index_loaded,
        "document_count": doc_count,
    }

