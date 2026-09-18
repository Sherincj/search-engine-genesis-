"""
URL frontier — breadth-first URL queue with duplicate prevention.

The frontier manages which URLs to crawl next, tracking depth
and ensuring each URL is visited at most once per crawl.
"""

from __future__ import annotations

from collections import deque

from app.crawler.models import CrawlTarget


class URLFrontier:
    """
    A BFS URL queue with duplicate prevention.

    Usage::

        frontier = URLFrontier()
        frontier.add("https://example.com", depth=0)
        target = frontier.pop()  # CrawlTarget(url=..., depth=0)
    """

    def __init__(self) -> None:
        self._queue: deque[CrawlTarget] = deque()
        self._seen: set[str] = set()

    def add(self, url: str, depth: int) -> bool:
        """
        Add a URL to the frontier.

        Returns ``True`` if the URL was newly added,
        ``False`` if it was already seen.
        """
        if url in self._seen:
            return False
        self._seen.add(url)
        self._queue.append(CrawlTarget(url=url, depth=depth))
        return True

    def pop(self) -> CrawlTarget | None:
        """
        Remove and return the next URL to crawl.

        Returns ``None`` if the frontier is empty.
        """
        if not self._queue:
            return None
        return self._queue.popleft()

    def __len__(self) -> int:
        """Return the number of URLs waiting to be crawled."""
        return len(self._queue)

    def empty(self) -> bool:
        """Whether the frontier has no more URLs to process."""
        return len(self._queue) == 0

    @property
    def seen_count(self) -> int:
        """Total number of unique URLs that have been added."""
        return len(self._seen)
