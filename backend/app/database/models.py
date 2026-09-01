"""
SQLAlchemy declarative base and ORM models.

Phase 1: Only the Base is defined.  Tables will be added in Phase 2
(documents, crawled_urls, links, crawl_jobs).
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass
