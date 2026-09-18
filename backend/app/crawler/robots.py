"""
Robots.txt policy — determines which URLs the crawler may fetch.

Uses Python's built-in urllib.robotparser for parsing.
Caches policies per origin to avoid re-fetching.

Failure policy:
- 404: treat as no restrictions (allow all).
- successful response: parse and enforce rules.
- timeout / network error: block crawling for that host.
"""

from __future__ import annotations

import urllib.robotparser
from urllib.parse import urlparse

import httpx

from app.utils.logging import get_logger

logger = get_logger(__name__)


class RobotsPolicy:
    """
    Checks robots.txt rules and caches them per origin.

    Parameters
    ----------
    user_agent : str
        The user-agent string to check rules against.
    timeout : float
        HTTP timeout for fetching robots.txt files.
    """

    def __init__(
        self,
        user_agent: str = "*",
        timeout: float = 10.0,
    ) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        # Cache: origin -> (RobotFileParser | None)
        # None means the robots.txt could not be fetched (block all).
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _get_origin(self, url: str) -> str:
        """Extract the origin (scheme + host + port) from a URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def is_allowed(self, url: str) -> bool:
        """
        Check if the configured user-agent may fetch *url*.

        Fetches and caches robots.txt for the origin on first call.

        Parameters
        ----------
        url : str
            The URL to check.

        Returns
        -------
        bool
            Whether the crawler may fetch this URL.
        """
        origin = self._get_origin(url)

        if origin not in self._cache:
            await self._fetch_robots(origin)

        parser = self._cache.get(origin)
        if parser is None:
            # Could not fetch robots.txt — conservative: block.
            return False

        return parser.can_fetch(self._user_agent, url)

    async def _fetch_robots(self, origin: str) -> None:
        """Fetch and cache robots.txt for the given origin."""
        robots_url = f"{origin}/robots.txt"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    robots_url,
                    timeout=self._timeout,
                    follow_redirects=True,
                )

            if response.status_code == 404:
                # No robots.txt — allow everything.
                parser = urllib.robotparser.RobotFileParser()
                parser.parse([])  # Empty rules = allow all.
                self._cache[origin] = parser
                logger.info("robots.txt not found for %s — allowing all", origin)

            elif 200 <= response.status_code < 300:
                parser = urllib.robotparser.RobotFileParser()
                content = response.text
                parser.parse(content.splitlines())
                self._cache[origin] = parser
                logger.info("robots.txt loaded for %s", origin)

            else:
                # Non-404 error — conservative: block.
                self._cache[origin] = None
                logger.warning(
                    "robots.txt returned %d for %s — blocking",
                    response.status_code,
                    origin,
                )

        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as exc:
            # Network failure — conservative: block.
            self._cache[origin] = None
            logger.warning(
                "Failed to fetch robots.txt for %s: %s — blocking",
                origin,
                exc,
            )

    def clear_cache(self) -> None:
        """Clear cached robots policies."""
        self._cache.clear()
