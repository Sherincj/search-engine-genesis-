"""
Health-check endpoint.

Returns basic system status so operators (and the frontend) can
confirm the backend is alive and which capabilities are ready.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Return current system health.

    Later phases will populate ``index_loaded`` and ``document_count``
    from real data.
    """
    return {
        "status": "ok",
        "index_loaded": False,
        "document_count": 0,
    }
