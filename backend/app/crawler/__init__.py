"""
Crawler package — web crawling and document ingestion.

Phase 10: HTTP fetching, URL normalization, robots.txt,
HTML parsing, and breadth-first web crawling.
"""

from app.crawler.crawler import Crawler
from app.crawler.fetcher import Fetcher
from app.crawler.frontier import URLFrontier
from app.crawler.models import CrawlResult, CrawlStats, CrawlTarget, FetchResult, ParsedPage
from app.crawler.parser import parse_html
from app.crawler.robots import RobotsPolicy
from app.crawler.url_normalizer import (
    extract_domain,
    is_same_domain,
    is_valid_url,
    normalize_url,
)

__all__ = [
    "Crawler",
    "CrawlResult",
    "CrawlStats",
    "CrawlTarget",
    "Fetcher",
    "FetchResult",
    "ParsedPage",
    "RobotsPolicy",
    "URLFrontier",
    "extract_domain",
    "is_same_domain",
    "is_valid_url",
    "normalize_url",
    "parse_html",
]
