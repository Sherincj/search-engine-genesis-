"""
Repository layer — data-access abstractions.

API routes call the repository; the repository talks to SQLAlchemy.
This keeps database queries out of the API layer.

Architecture::

    API Route
       ↓
    DocumentRepository
       ↓
    SQLAlchemy AsyncSession
       ↓
    SQLite
"""

import hashlib
from urllib.parse import urlparse

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Document


class DuplicateURLError(Exception):
    """Raised when a document with the same URL already exists."""

    pass


class DocumentRepository:
    """
    Encapsulates all database operations for Document records.

    Usage::

        repo = DocumentRepository(session)
        doc = await repo.create_document(url=..., title=..., text=...)
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _extract_domain(url: str) -> str:
        """
        Derive the domain from a URL, stripping the ``www.`` prefix.

        Example::

            https://www.example.com/articles/python  →  example.com
        """
        hostname = urlparse(url).hostname or ""
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname.lower()

    @staticmethod
    def _calculate_content_hash(text: str) -> str:
        """Return the SHA-256 hex-digest of the document text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _calculate_word_count(text: str) -> int:
        """
        Simple whitespace-based word count.

        A proper tokeniser will replace this in Phase 3.  For now a
        ``str.split()`` count is good enough for the stored metadata.
        """
        return len(text.split())

    # ── CRUD ─────────────────────────────────────────────────

    async def create_document(
        self,
        *,
        url: str,
        title: str,
        text: str,
        canonical_url: str | None = None,
    ) -> Document:
        """
        Create and persist a new document.

        Derives domain, content_hash, and word_count automatically.
        Raises ``DuplicateURLError`` if the URL already exists.
        """
        doc = Document(
            url=url,
            canonical_url=canonical_url,
            domain=self._extract_domain(url),
            title=title,
            text=text,
            content_hash=self._calculate_content_hash(text),
            word_count=self._calculate_word_count(text),
            status="stored",
        )

        self._session.add(doc)

        try:
            await self._session.commit()
        except IntegrityError:
            await self._session.rollback()
            raise DuplicateURLError(
                f"A document with URL '{url}' already exists."
            )

        await self._session.refresh(doc)
        return doc

    async def get_document(self, document_id: int) -> Document | None:
        """Return a document by primary key, or ``None`` if not found."""
        return await self._session.get(Document, document_id)

    async def delete_document(self, document_id: int) -> bool:
        """
        Delete a document by ID.

        Returns ``True`` if the document existed and was deleted,
        ``False`` if it was not found.
        """
        doc = await self._session.get(Document, document_id)
        if doc is None:
            return False
        await self._session.delete(doc)
        await self._session.commit()
        return True

    async def count_documents(self) -> int:
        """Return the total number of stored documents."""
        result = await self._session.execute(
            select(func.count()).select_from(Document)
        )
        return result.scalar_one()

    async def find_by_content_hash(self, content_hash: str) -> list[Document]:
        """
        Return all documents sharing the given content hash.

        Not exposed as a public API endpoint in Phase 2, but available
        for future duplicate-content detection by the indexer/crawler.
        """
        result = await self._session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        return list(result.scalars().all())
