"""
Crawl API — triggers web crawling and returns results.

Phase 10: POST /crawl to start a synchronous crawl.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.crawler.crawler import Crawler
from app.crawler.url_normalizer import is_valid_url
from app.database.database import get_db
from app.services.search_service import get_search_service
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/crawl", tags=["crawl"])


# ── Request / Response models ────────────────────────────────

class CrawlRequest(BaseModel):
    """Request body for starting a crawl."""

    url: str
    max_pages: int | None = None
    max_depth: int | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL must not be empty")
        if not is_valid_url(v):
            raise ValueError(
                f"Invalid or unsupported URL: {v}. "
                "Only http:// and https:// URLs are allowed."
            )
        return v


class CrawlResponse(BaseModel):
    """Summary of a completed crawl."""

    seed_url: str
    pages_crawled: int
    pages_stored: int
    pages_failed: int
    pages_skipped: int
    discovered_urls: int
    max_depth_reached: int


# ── Endpoint ─────────────────────────────────────────────────

@router.post("", response_model=CrawlResponse)
async def start_crawl(
    request: CrawlRequest,
    db: AsyncSession = Depends(get_db),
) -> CrawlResponse:
    """
    Start a crawl from the given seed URL.

    Crawls same-domain pages breadth-first, stores documents in
    SQLite, and updates the search index.
    """
    service = get_search_service()

    crawler = Crawler(
        db=db,
        index=service.index,
        # fetcher and robots use defaults from settings.
    )

    logger.info("Crawl API request: url=%s", request.url)

    result = await crawler.crawl(
        request.url,
        max_pages=request.max_pages,
        max_depth=request.max_depth,
    )

    # Persist the updated index to disk.
    service._storage.save(service.index)

    logger.info(
        "Crawl API completed: stored=%d crawled=%d",
        result.pages_stored,
        result.pages_crawled,
    )

    return CrawlResponse(
        seed_url=result.seed_url,
        pages_crawled=result.pages_crawled,
        pages_stored=result.pages_stored,
        pages_failed=result.pages_failed,
        pages_skipped=result.pages_skipped,
        discovered_urls=result.discovered_urls,
        max_depth_reached=result.max_depth_reached,
    )
