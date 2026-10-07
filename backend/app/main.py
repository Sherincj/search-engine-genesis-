"""
FastAPI application entry-point.

Creates the app, wires up middleware, registers routers,
and manages the database lifecycle via the lifespan handler.

Run with::

    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.database.database import init_db, close_db
from app.utils.logging import setup_logging, get_logger

# ── Routers ──────────────────────────────────────────────────
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.api.semantic_search import router as semantic_router
from app.api.hybrid_search import router as hybrid_router
from app.api.documents import router as documents_router
from app.api.crawl import router as crawl_router

logger = get_logger(__name__)


# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle for the application."""
    setup_logging(debug=settings.DEBUG)
    logger.info("Starting Search Engine backend …")
    await init_db()
    logger.info("Database initialised.")
    yield
    await close_db()
    logger.info("Database closed.  Goodbye.")


# ── App factory ──────────────────────────────────────────────
app = FastAPI(
    title="Search Engine API",
    description="A search engine built from scratch.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js dev server to talk to us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────
app.include_router(health_router)
app.include_router(search_router)
app.include_router(semantic_router)
app.include_router(hybrid_router)
app.include_router(documents_router)
app.include_router(crawl_router)
