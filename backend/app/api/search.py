"""
Search API — placeholder router.

The actual search endpoint will be implemented in Phase 5+.
For now, every route returns 501 Not Implemented so the API
surface is discoverable but clearly unfinished.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
async def search(q: str = "", page: int = 1, limit: int = 10) -> JSONResponse:
    """Search the index. Not yet implemented."""
    return JSONResponse(
        status_code=501,
        content={"detail": "Search not implemented yet (Phase 5)."},
    )
