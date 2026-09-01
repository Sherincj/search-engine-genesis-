"""
Crawl API — placeholder router.

The actual crawl endpoints will be implemented in Phase 10.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/crawl", tags=["crawl"])


@router.post("")
async def start_crawl() -> JSONResponse:
    """Start a crawl job. Not yet implemented."""
    return JSONResponse(
        status_code=501,
        content={"detail": "Crawl not implemented yet (Phase 10)."},
    )


@router.get("/{job_id}")
async def get_crawl_status(job_id: str) -> JSONResponse:
    """Get crawl job status. Not yet implemented."""
    return JSONResponse(
        status_code=501,
        content={"detail": "Crawl status not implemented yet (Phase 10)."},
    )
