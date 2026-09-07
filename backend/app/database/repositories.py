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

Note: word_count uses the Phase 3 tokenizer for accurate counting.
Documents created under Phase 2 used ``text.split()``; new documents
use tokenizer-based counting and may differ for punctuation-heavy text.
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
        Token-based word count using the Phase 3 tokeniser.

        Uses default config (lowercase, Unicode normalisation,
        stop-word removal **off**).  Replaces the Phase 2
        whitespace-based ``text.split()`` implementation.
        """
        from app.text_processing import tokenize

        return len(tokenize(text))

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

    async def get_by_ids(self, document_ids: list[int]) -> list[Document]:
        """
        Retrieve multiple documents by ID in a single query.

        The returned list preserves the order of *document_ids*.
        IDs not found in the database are silently skipped.

        Parameters
        ----------
        document_ids : list[int]
            Ordered list of document IDs to fetch.

        Returns
        -------
        list[Document]
            Documents in the same order as *document_ids*.
        """
        if not document_ids:
            return []

        result = await self._session.execute(
            select(Document).where(Document.id.in_(document_ids))
        )
        docs_by_id = {doc.id: doc for doc in result.scalars().all()}

        # Preserve caller's ordering; skip missing IDs.
        return [docs_by_id[did] for did in document_ids if did in docs_by_id]

    async def get_all_documents(self) -> list[Document]:
        """Return all stored documents (for index rebuilding)."""
        result = await self._session.execute(select(Document))
        return list(result.scalars().all())

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
