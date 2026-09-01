"""
Database engine, session factory, and lifecycle helpers.

Uses SQLAlchemy 2.0 async API with aiosqlite for SQLite.
The engine is created lazily on first call to ``init_db()``.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import settings
from app.database.models import Base

# ── Module-level state (initialised via lifespan) ────────────
_engine = None
_session_factory = None


async def init_db() -> None:
    """
    Create the async engine, session factory, and all tables.

    Called once during application startup (FastAPI lifespan).
    """
    global _engine, _session_factory

    # Ensure the directory for the SQLite file exists.
    db_path = settings.DATABASE_URL.split("///")[-1]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        future=True,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables that don't yet exist.
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the engine.  Called during application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.

    Usage in a route::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    async with _session_factory() as session:
        yield session
