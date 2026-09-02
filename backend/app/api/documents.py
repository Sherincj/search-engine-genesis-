"""
Document API — CRUD endpoints for document storage.

Architecture::

    POST /documents      →  create
    GET  /documents/{id} →  retrieve
    DELETE /documents/{id} →  delete

All database access goes through DocumentRepository;
no raw SQLAlchemy queries live in this file.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.database.repositories import DocumentRepository, DuplicateURLError
from app.models.document import DocumentCreate, DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])


# ── POST /documents ──────────────────────────────────────────

@router.post("", response_model=DocumentResponse, status_code=201)
async def create_document(
    body: DocumentCreate,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """
    Store a new document.

    The backend derives domain, word_count, content_hash, timestamps,
    and status automatically.  Returns 409 if the URL already exists.
    """
    repo = DocumentRepository(db)

    try:
        doc = await repo.create_document(
            url=body.url,
            title=body.title,
            text=body.text,
            canonical_url=body.canonical_url,
        )
    except DuplicateURLError:
        raise HTTPException(
            status_code=409,
            detail="A document with this URL already exists.",
        )

    return DocumentResponse.model_validate(doc)


# ── GET /documents/{document_id} ─────────────────────────────

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Retrieve a single document by ID.  Returns 404 if not found."""
    repo = DocumentRepository(db)
    doc = await repo.get_document(document_id)

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentResponse.model_validate(doc)


# ── DELETE /documents/{document_id} ──────────────────────────

@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document by ID.  Returns 204 on success, 404 if not found."""
    repo = DocumentRepository(db)
    deleted = await repo.delete_document(document_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
