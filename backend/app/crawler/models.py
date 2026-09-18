"""
Crawler domain models — internal types for the crawl pipeline.

These types flow through the crawler but are NOT directly exposed
as API response models.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrawlTarget:
    """A URL queued for crawling with its depth."""
    url: str
    depth: int


@dataclass
class FetchResult:
    """Result of fetching a single URL."""
    url: str
    final_url: str
    status_code: int
    content_type: str | None
    content: bytes
    error: str | None = None

    @property
    def is_html(self) -> bool:
        """Whether the response content type is HTML."""
        if not self.content_type:
            return False
        ct = self.content_type.lower().split(";")[0].strip()
        return ct in ("text/html", "application/xhtml+xml")

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@dataclass
class ParsedPage:
    """Result of parsing an HTML page."""
    title: str | None
    text: str
    links: list[str]
    canonical_url: str | None = None


@dataclass
class CrawlResult:
    """Summary of a completed crawl."""
    seed_url: str
    pages_crawled: int = 0
    pages_stored: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    discovered_urls: int = 0
    max_depth_reached: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class CrawlStats:
    """Detailed crawl statistics for internal tracking."""
    queued: int = 0
    fetched: int = 0
    stored: int = 0
    failed: int = 0
    skipped: int = 0
    robots_blocked: int = 0
    non_html: int = 0
    duplicate: int = 0
