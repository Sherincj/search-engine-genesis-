"""
SQLAlchemy declarative base and ORM models.

Phase 1 — Base class only.
Phase 2 — Document model for persistent document storage.

Future phases will add CrawledURL, Link, and CrawlJob models.
"""

from datetime import datetime, timezone

from sqlalchemy import Index, Integer, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Document(Base):
    """
    A stored document — the fundamental unit of the search engine.

    Stores the raw text, metadata, and derived fields (domain, content_hash,
    word_count) for every page that the system knows about.  Later phases
    will tokenise, index, and rank these documents.
    """

    __tablename__ = "documents"

    # ── Primary key ──────────────────────────────────────────
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Source identifiers ───────────────────────────────────
    url: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    canonical_url: Mapped[str | None] = mapped_column(String, nullable=True)
    domain: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # ── Content ──────────────────────────────────────────────
    title: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Derived metadata ─────────────────────────────────────
    content_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Timestamps (always UTC) ──────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # ── Lifecycle state ──────────────────────────────────────
    # Phase 2 uses only "stored".  Future phases may add
    # "indexing", "indexed", "failed".
    status: Mapped[str] = mapped_column(String, nullable=False, default="stored")

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, url={self.url!r}, title={self.title!r})>"
