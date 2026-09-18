"""
Web crawler — orchestrates URL frontier, fetcher, parser, robots,
document storage, and index updates.

Phase 10: single-threaded, breadth-first, same-domain crawling.

Architecture::

    Crawler
        ├── URLFrontier (BFS queue, dedup)
        ├── RobotsPolicy (per-origin caching)
        ├── Fetcher (httpx)
        ├── parse_html (BeautifulSoup)
        ├── DocumentRepository (SQLite)
        └── DocumentIndexer (FieldIndex)
"""

from __future__ import annotations

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.crawler.fetcher import Fetcher
from app.crawler.frontier import URLFrontier
from app.crawler.models import CrawlResult, CrawlStats, ParsedPage
from app.crawler.parser import parse_html
from app.crawler.robots import RobotsPolicy
from app.crawler.url_normalizer import (
    extract_domain,
    is_same_domain,
    is_valid_url,
    normalize_url,
)
from app.database.repositories import DocumentRepository, DuplicateURLError
from app.indexing import DocumentIndexer, FieldIndex
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Crawler:
    """
    Breadth-first, same-domain web crawler.

    Parameters
    ----------
    db : AsyncSession
        Database session for document persistence.
    index : FieldIndex
        The field-aware index to add crawled documents to.
    fetcher : Fetcher | None
        HTTP fetcher (injected for testing).
    robots : RobotsPolicy | None
        Robots policy (injected for testing).
    """

    def __init__(
        self,
        db: AsyncSession,
        index: FieldIndex,
        *,
        fetcher: Fetcher | None = None,
        robots: RobotsPolicy | None = None,
    ) -> None:
        self._db = db
        self._index = index
        self._indexer = DocumentIndexer(index)
        self._repo = DocumentRepository(db)
        self._fetcher = fetcher or Fetcher(
            user_agent=settings.CRAWLER_USER_AGENT,
            timeout=settings.CRAWLER_TIMEOUT,
            max_response_bytes=settings.CRAWLER_MAX_RESPONSE_BYTES,
        )
        self._robots = robots or RobotsPolicy(
            user_agent=settings.CRAWLER_USER_AGENT,
            timeout=settings.CRAWLER_TIMEOUT,
        )

    async def crawl(
        self,
        seed_url: str,
        *,
        max_pages: int | None = None,
        max_depth: int | None = None,
    ) -> CrawlResult:
        """
        Execute a breadth-first crawl starting from *seed_url*.

        Parameters
        ----------
        seed_url : str
            Starting URL.
        max_pages : int | None
            Maximum pages to successfully process.
            Defaults to ``settings.MAX_CRAWL_PAGES``.
        max_depth : int | None
            Maximum link depth from the seed.
            Defaults to ``settings.MAX_CRAWL_DEPTH``.

        Returns
        -------
        CrawlResult
            Summary of the completed crawl.
        """
        if max_pages is None:
            max_pages = settings.MAX_CRAWL_PAGES
        if max_depth is None:
            max_depth = settings.MAX_CRAWL_DEPTH

        # 1. Normalize and validate seed URL.
        normalized_seed = normalize_url(seed_url)
        if not normalized_seed:
            return CrawlResult(
                seed_url=seed_url,
                errors=[f"Invalid seed URL: {seed_url}"],
            )

        result = CrawlResult(seed_url=normalized_seed)
        stats = CrawlStats()

        # 2. Initialize frontier.
        frontier = URLFrontier()
        frontier.add(normalized_seed, depth=0)
        stats.queued += 1

        logger.info("Crawl started: seed=%s max_pages=%d max_depth=%d",
                     normalized_seed, max_pages, max_depth)

        # 3. BFS loop.
        while not frontier.empty() and result.pages_stored < max_pages:
            target = frontier.pop()
            if target is None:
                break

            url = target.url
            depth = target.depth

            # ── Robots check ─────────────────────────────────
            if not await self._robots.is_allowed(url):
                logger.debug("Robots blocked: %s", url)
                stats.robots_blocked += 1
                result.pages_skipped += 1
                continue

            # ── Fetch ─────────────────────────────────────────
            fetch_result = await self._fetcher.fetch(url)
            stats.fetched += 1

            if fetch_result.error:
                logger.warning("Fetch failed: %s — %s", url, fetch_result.error)
                result.pages_failed += 1
                result.errors.append(f"{url}: {fetch_result.error}")
                continue

            if not fetch_result.ok:
                logger.debug("Non-OK status %d: %s", fetch_result.status_code, url)
                result.pages_failed += 1
                result.errors.append(
                    f"{url}: HTTP {fetch_result.status_code}"
                )
                continue

            # ── Content-type check ────────────────────────────
            if not fetch_result.is_html:
                logger.debug("Non-HTML content: %s", url)
                stats.non_html += 1
                result.pages_skipped += 1
                continue

            result.pages_crawled += 1

            # ── Parse HTML ────────────────────────────────────
            html = fetch_result.content.decode("utf-8", errors="replace")
            final_url = normalize_url(fetch_result.final_url) or url
            parsed = parse_html(html, base_url=final_url)

            # ── Store document ────────────────────────────────
            title = parsed.title or ""
            text = parsed.text or ""

            # Use canonical URL if valid and same-domain.
            doc_url = final_url
            if parsed.canonical_url:
                canon = normalize_url(parsed.canonical_url, base_url=final_url)
                if canon and is_same_domain(canon, normalized_seed):
                    doc_url = canon

            try:
                doc = await self._repo.create_document(
                    url=doc_url,
                    title=title,
                    text=text,
                    canonical_url=parsed.canonical_url,
                )
                result.pages_stored += 1
                stats.stored += 1

                # ── Index the document ────────────────────────
                self._indexer.index_document(
                    doc.id, text, title=title,
                )
                logger.info("Stored and indexed: %s (id=%d)", doc_url, doc.id)

            except DuplicateURLError:
                logger.debug("Duplicate URL skipped: %s", doc_url)
                stats.duplicate += 1
                result.pages_skipped += 1

            # ── Extract and enqueue links ─────────────────────
            if depth < max_depth:
                for link in parsed.links:
                    new_url = normalize_url(link, base_url=final_url)
                    if not new_url:
                        continue
                    if not is_same_domain(new_url, normalized_seed):
                        continue
                    if frontier.add(new_url, depth=depth + 1):
                        stats.queued += 1

            # Track max depth reached.
            if depth > result.max_depth_reached:
                result.max_depth_reached = depth

            # ── Crawl delay ───────────────────────────────────
            if not frontier.empty() and result.pages_stored < max_pages:
                await asyncio.sleep(settings.CRAWLER_DELAY)

        result.discovered_urls = frontier.seen_count

        logger.info(
            "Crawl completed: seed=%s stored=%d crawled=%d failed=%d "
            "discovered=%d depth=%d",
            normalized_seed,
            result.pages_stored,
            result.pages_crawled,
            result.pages_failed,
            result.discovered_urls,
            result.max_depth_reached,
        )

        return result
