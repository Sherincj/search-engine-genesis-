"""
Document API — placeholder router.

The actual document CRUD will be implemented in Phase 2.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("")
async def create_document() -> JSONResponse:
    """Create a document. Not yet implemented."""
    return JSONResponse(
        status_code=501,
        content={"detail": "Document creation not implemented yet (Phase 2)."},
    )


@router.get("/{document_id}")
async def get_document(document_id: int) -> JSONResponse:
    """Retrieve a document by ID. Not yet implemented."""
    return JSONResponse(
        status_code=501,
        content={"detail": "Document retrieval not implemented yet (Phase 2)."},
    )


@router.delete("/{document_id}")
async def delete_document(document_id: int) -> JSONResponse:
    """Delete a document by ID. Not yet implemented."""
    return JSONResponse(
        status_code=501,
        content={"detail": "Document deletion not implemented yet (Phase 2)."},
    )
