"""
HTTP fetcher — responsible only for retrieving web pages.

Does NOT:
- Parse HTML
- Extract links
- Write to database
- Manage crawl queues
"""

from __future__ import annotations

import httpx

from app.crawler.models import FetchResult
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Default maximum response size (5 MB).
DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class Fetcher:
    """
    Async HTTP fetcher for crawling.

    Parameters
    ----------
    user_agent : str
        User-Agent header to send.
    timeout : float
        Request timeout in seconds.
    max_response_bytes : int
        Maximum response body size.
    max_redirects : int
        Maximum redirect hops to follow.
    """

    def __init__(
        self,
        user_agent: str = "SearchEngineBot/1.0",
        timeout: float = 10.0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_redirects: int = 10,
    ) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._max_redirects = max_redirects

    async def fetch(self, url: str) -> FetchResult:
        """
        Fetch a URL and return the result.

        Handles timeouts, connection errors, oversized responses,
        and HTTP error status codes gracefully.
        """
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=self._max_redirects,
                timeout=self._timeout,
            ) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self._user_agent},
                )

                # Check response size via Content-Length header.
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self._max_response_bytes:
                    return FetchResult(
                        url=url,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=response.headers.get("content-type"),
                        content=b"",
                        error=f"Response too large: {content_length} bytes",
                    )

                content = response.content
                if len(content) > self._max_response_bytes:
                    return FetchResult(
                        url=url,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=response.headers.get("content-type"),
                        content=b"",
                        error=f"Response too large: {len(content)} bytes",
                    )

                return FetchResult(
                    url=url,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type"),
                    content=content,
                )

        except httpx.TimeoutException:
            logger.warning("Timeout fetching %s", url)
            return FetchResult(
                url=url, final_url=url, status_code=0,
                content_type=None, content=b"",
                error="Request timed out",
            )

        except httpx.TooManyRedirects:
            logger.warning("Too many redirects for %s", url)
            return FetchResult(
                url=url, final_url=url, status_code=0,
                content_type=None, content=b"",
                error="Too many redirects",
            )

        except httpx.ConnectError as exc:
            logger.warning("Connection error for %s: %s", url, exc)
            return FetchResult(
                url=url, final_url=url, status_code=0,
                content_type=None, content=b"",
                error=f"Connection error: {exc}",
            )

        except httpx.HTTPError as exc:
            logger.warning("HTTP error for %s: %s", url, exc)
            return FetchResult(
                url=url, final_url=url, status_code=0,
                content_type=None, content=b"",
                error=f"HTTP error: {exc}",
            )
