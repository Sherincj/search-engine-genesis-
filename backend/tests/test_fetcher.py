"""Tests for the HTTP fetcher."""

from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from app.crawler.fetcher import Fetcher


# ── Helpers ──────────────────────────────────────────────────

def _make_response(
    status_code: int = 200,
    content: bytes = b"<html><body>OK</body></html>",
    content_type: str = "text/html",
    url: str = "https://example.com",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = {"content-type": content_type}
    resp.url = httpx.URL(url)
    return resp


def _patch_client(response=None, side_effect=None):
    mock_client = AsyncMock()
    if side_effect:
        mock_client.get = AsyncMock(side_effect=side_effect)
    else:
        mock_client.get = AsyncMock(return_value=response or _make_response())
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return patch("app.crawler.fetcher.httpx.AsyncClient", return_value=mock_client)


# ── Tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_fetch():
    with _patch_client():
        fetcher = Fetcher()
        result = await fetcher.fetch("https://example.com")
        assert result.ok
        assert result.status_code == 200
        assert result.is_html
        assert result.error is None


@pytest.mark.asyncio
async def test_timeout():
    with _patch_client(side_effect=httpx.TimeoutException("timeout")):
        fetcher = Fetcher()
        result = await fetcher.fetch("https://example.com")
        assert not result.ok
        assert result.error is not None
        assert "timed out" in result.error


@pytest.mark.asyncio
async def test_connection_error():
    with _patch_client(side_effect=httpx.ConnectError("refused")):
        fetcher = Fetcher()
        result = await fetcher.fetch("https://example.com")
        assert not result.ok
        assert result.error is not None
        assert "Connection error" in result.error


@pytest.mark.asyncio
async def test_404():
    with _patch_client(_make_response(status_code=404)):
        fetcher = Fetcher()
        result = await fetcher.fetch("https://example.com/missing")
        assert result.status_code == 404
        assert not result.ok


@pytest.mark.asyncio
async def test_500():
    with _patch_client(_make_response(status_code=500)):
        fetcher = Fetcher()
        result = await fetcher.fetch("https://example.com")
        assert result.status_code == 500
        assert not result.ok


@pytest.mark.asyncio
async def test_redirect_final_url():
    resp = _make_response(url="https://www.example.com/final")
    with _patch_client(resp):
        fetcher = Fetcher()
        result = await fetcher.fetch("https://example.com")
        assert result.final_url == "https://www.example.com/final"


@pytest.mark.asyncio
async def test_non_html_content_type():
    resp = _make_response(content_type="application/pdf")
    with _patch_client(resp):
        fetcher = Fetcher()
        result = await fetcher.fetch("https://example.com/file.pdf")
        assert not result.is_html


@pytest.mark.asyncio
async def test_response_too_large():
    big_content = b"x" * (6 * 1024 * 1024)  # 6 MB
    resp = _make_response(content=big_content)
    with _patch_client(resp):
        fetcher = Fetcher(max_response_bytes=5 * 1024 * 1024)
        result = await fetcher.fetch("https://example.com")
        assert result.error is not None
        assert "too large" in result.error.lower()


@pytest.mark.asyncio
async def test_user_agent_sent():
    with _patch_client() as mock_cls:
        fetcher = Fetcher(user_agent="MyBot/1.0")
        await fetcher.fetch("https://example.com")
        client = mock_cls.return_value
        call_args = client.get.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers", {})
        assert headers.get("User-Agent") == "MyBot/1.0"


@pytest.mark.asyncio
async def test_too_many_redirects():
    with _patch_client(side_effect=httpx.TooManyRedirects("too many")):
        fetcher = Fetcher()
        result = await fetcher.fetch("https://example.com")
        assert not result.ok
        assert result.error is not None
        assert "redirect" in result.error.lower()
