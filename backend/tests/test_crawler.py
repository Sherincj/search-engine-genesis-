"""
Tests for the web crawler.

Uses a mock fetcher and robots policy to simulate a fake website graph
without hitting the real internet.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.crawler.crawler import Crawler
from app.crawler.fetcher import Fetcher
from app.crawler.models import FetchResult
from app.crawler.robots import RobotsPolicy
from app.database.models import Base, Document
from app.database.repositories import DocumentRepository
from app.indexing import FieldIndex


# ── Fake website builder ─────────────────────────────────────

def _html_page(
    title: str,
    body: str,
    links: list[str] | None = None,
    canonical: str | None = None,
) -> str:
    link_tags = ""
    if links:
        link_tags = "".join(f'<a href="{u}">Link</a>' for u in links)
    canon_tag = ""
    if canonical:
        canon_tag = f'<link rel="canonical" href="{canonical}">'
    return f"""
    <html>
    <head><title>{title}</title>{canon_tag}</head>
    <body>
        <p>{body}</p>
        {link_tags}
    </body>
    </html>
    """


class FakeFetcher(Fetcher):
    """A fetcher that serves pages from a dict instead of the internet."""

    def __init__(self, pages: dict[str, str | None]):
        """pages: {url: html_content_or_None}"""
        super().__init__()
        self._pages = pages

    async def fetch(self, url: str) -> FetchResult:
        if url in self._pages and self._pages[url] is not None:
            content = self._pages[url].encode("utf-8")
            return FetchResult(
                url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                content=content,
            )
        return FetchResult(
            url=url,
            final_url=url,
            status_code=404,
            content_type=None,
            content=b"",
            error="Not found",
        )


class AllowAllRobots(RobotsPolicy):
    """A robots policy that allows everything."""

    def __init__(self):
        super().__init__()

    async def is_allowed(self, url: str) -> bool:
        return True


class BlockAllRobots(RobotsPolicy):
    """A robots policy that blocks everything."""

    def __init__(self):
        super().__init__()

    async def is_allowed(self, url: str) -> bool:
        return False


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
async def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

    await engine.dispose()


# ── Crawl tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_basic_crawl(db_session):
    """Crawl a single page."""
    pages = {
        "https://example.com/": _html_page(
            "Example", "Hello World",
        ),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )
    result = await crawler.crawl("https://example.com/", max_pages=10, max_depth=0)

    assert result.pages_stored == 1
    assert result.pages_crawled == 1
    assert result.pages_failed == 0


@pytest.mark.asyncio
async def test_crawl_follows_links(db_session):
    """Crawl follows same-domain links."""
    pages = {
        "https://example.com/": _html_page(
            "Home", "Homepage",
            links=["/about", "/contact"],
        ),
        "https://example.com/about": _html_page(
            "About", "About page",
        ),
        "https://example.com/contact": _html_page(
            "Contact", "Contact page",
        ),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )
    result = await crawler.crawl(
        "https://example.com/", max_pages=10, max_depth=1,
    )

    assert result.pages_stored == 3
    assert result.pages_crawled == 3


@pytest.mark.asyncio
async def test_duplicate_urls_crawled_once(db_session):
    """Same URL linked multiple times is only crawled once."""
    pages = {
        "https://example.com/": _html_page(
            "Home", "Homepage",
            links=["/about", "/about", "/about"],
        ),
        "https://example.com/about": _html_page(
            "About", "About page",
        ),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )
    result = await crawler.crawl(
        "https://example.com/", max_pages=10, max_depth=1,
    )

    assert result.pages_stored == 2


@pytest.mark.asyncio
async def test_depth_limit(db_session):
    """Depth limit prevents crawling deeper pages."""
    pages = {
        "https://example.com/": _html_page("A", "Page A", links=["/b"]),
        "https://example.com/b": _html_page("B", "Page B", links=["/c"]),
        "https://example.com/c": _html_page("C", "Page C", links=["/d"]),
        "https://example.com/d": _html_page("D", "Page D"),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )

    # max_depth=1 should crawl A (depth 0) and B (depth 1) but not C or D.
    result = await crawler.crawl(
        "https://example.com/", max_pages=10, max_depth=1,
    )

    assert result.pages_stored == 2


@pytest.mark.asyncio
async def test_page_limit(db_session):
    """Page limit stops crawling early."""
    pages = {
        "https://example.com/": _html_page("A", "Page A", links=["/b", "/c", "/d"]),
        "https://example.com/b": _html_page("B", "Page B"),
        "https://example.com/c": _html_page("C", "Page C"),
        "https://example.com/d": _html_page("D", "Page D"),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )

    result = await crawler.crawl(
        "https://example.com/", max_pages=2, max_depth=2,
    )

    assert result.pages_stored == 2


@pytest.mark.asyncio
async def test_same_domain_only(db_session):
    """Cross-domain links are not followed."""
    pages = {
        "https://example.com/": _html_page(
            "Home", "Homepage",
            links=["https://other.com/page", "/local"],
        ),
        "https://example.com/local": _html_page("Local", "Local page"),
        "https://other.com/page": _html_page("Other", "Other domain"),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )

    result = await crawler.crawl(
        "https://example.com/", max_pages=10, max_depth=1,
    )

    assert result.pages_stored == 2  # only example.com pages


@pytest.mark.asyncio
async def test_robots_blocks(db_session):
    """Robots policy blocks crawling."""
    pages = {
        "https://example.com/": _html_page("Home", "Homepage"),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=BlockAllRobots(),
    )

    result = await crawler.crawl(
        "https://example.com/", max_pages=10, max_depth=0,
    )

    assert result.pages_stored == 0
    assert result.pages_skipped >= 1


@pytest.mark.asyncio
async def test_failed_fetch_continues(db_session):
    """Failed fetches don't crash the crawl."""
    pages = {
        "https://example.com/": _html_page(
            "Home", "Homepage", links=["/missing", "/good"],
        ),
        "https://example.com/good": _html_page("Good", "Good page"),
        # /missing returns 404
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )

    result = await crawler.crawl(
        "https://example.com/", max_pages=10, max_depth=1,
    )

    assert result.pages_stored >= 2  # Home + Good
    assert result.pages_failed >= 1  # /missing


@pytest.mark.asyncio
async def test_invalid_seed_url(db_session):
    """Invalid seed URL returns error immediately."""
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher({}),
        robots=AllowAllRobots(),
    )

    result = await crawler.crawl("not-a-url")

    assert result.pages_stored == 0
    assert len(result.errors) > 0


@pytest.mark.asyncio
async def test_depth_zero_seed_only(db_session):
    """max_depth=0 only crawls the seed."""
    pages = {
        "https://example.com/": _html_page("Home", "Homepage", links=["/about"]),
        "https://example.com/about": _html_page("About", "About page"),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )

    result = await crawler.crawl(
        "https://example.com/", max_pages=10, max_depth=0,
    )

    assert result.pages_stored == 1


@pytest.mark.asyncio
async def test_index_updated(db_session):
    """Crawled documents are added to the FieldIndex."""
    pages = {
        "https://example.com/": _html_page(
            "Python Guide", "Python is a programming language",
        ),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )

    await crawler.crawl("https://example.com/", max_pages=10, max_depth=0)

    assert index.get_total_documents() == 1
    # Title should be indexed.
    assert len(index.title_index.get_postings("python")) > 0
    # Body should be indexed.
    assert len(index.body_index.get_postings("python")) > 0


@pytest.mark.asyncio
async def test_crawl_result_stats(db_session):
    """CrawlResult contains correct statistics."""
    pages = {
        "https://example.com/": _html_page(
            "Home", "Homepage", links=["/a", "/b"],
        ),
        "https://example.com/a": _html_page("A", "Page A"),
        "https://example.com/b": _html_page("B", "Page B"),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )

    result = await crawler.crawl(
        "https://example.com/", max_pages=10, max_depth=1,
    )

    assert result.seed_url == "https://example.com/"
    assert result.pages_stored == 3
    assert result.pages_crawled == 3
    assert result.discovered_urls >= 3
    assert result.max_depth_reached >= 0


@pytest.mark.asyncio
async def test_search_after_crawl(db_session):
    """
    Integration: crawl → index → search.

    Proves that crawled documents are searchable.
    """
    pages = {
        "https://example.com/": _html_page(
            "Python Guide",
            "Python is a versatile programming language for web development",
        ),
    }
    index = FieldIndex()
    crawler = Crawler(
        db_session, index,
        fetcher=FakeFetcher(pages),
        robots=AllowAllRobots(),
    )

    await crawler.crawl("https://example.com/", max_pages=10, max_depth=0)

    # Search through the engine.
    from app.search.search_engine import SearchEngine

    engine = SearchEngine(index)
    result = engine.search("python")

    assert result.total >= 1
    assert result.ranked[0].score > 0
