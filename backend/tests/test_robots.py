"""Tests for the robots.txt policy handler."""

from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from app.crawler.robots import RobotsPolicy


# ── Helpers ──────────────────────────────────────────────────

def _mock_response(status_code: int = 200, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


def _patch_httpx(response):
    """Patch httpx.AsyncClient to return a mock response."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return patch("app.crawler.robots.httpx.AsyncClient", return_value=mock_client)


# ── Tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allows_url():
    robots_txt = "User-agent: *\nAllow: /"
    with _patch_httpx(_mock_response(200, robots_txt)):
        policy = RobotsPolicy(user_agent="TestBot")
        assert await policy.is_allowed("https://example.com/page") is True


@pytest.mark.asyncio
async def test_blocks_url():
    robots_txt = "User-agent: *\nDisallow: /private/"
    with _patch_httpx(_mock_response(200, robots_txt)):
        policy = RobotsPolicy(user_agent="TestBot")
        assert await policy.is_allowed("https://example.com/private/secret") is False


@pytest.mark.asyncio
async def test_404_allows_all():
    with _patch_httpx(_mock_response(404)):
        policy = RobotsPolicy(user_agent="TestBot")
        assert await policy.is_allowed("https://example.com/anything") is True


@pytest.mark.asyncio
async def test_fetch_failure_blocks():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("fail"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("app.crawler.robots.httpx.AsyncClient", return_value=mock_client):
        policy = RobotsPolicy(user_agent="TestBot")
        assert await policy.is_allowed("https://example.com/page") is False


@pytest.mark.asyncio
async def test_timeout_blocks():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("app.crawler.robots.httpx.AsyncClient", return_value=mock_client):
        policy = RobotsPolicy(user_agent="TestBot")
        assert await policy.is_allowed("https://example.com/page") is False


@pytest.mark.asyncio
async def test_cached():
    """robots.txt should only be fetched once per origin."""
    robots_txt = "User-agent: *\nAllow: /"
    resp = _mock_response(200, robots_txt)
    with _patch_httpx(resp) as mock_cls:
        policy = RobotsPolicy(user_agent="TestBot")
        await policy.is_allowed("https://example.com/page1")
        await policy.is_allowed("https://example.com/page2")
        client = mock_cls.return_value
        assert client.get.await_count == 1


@pytest.mark.asyncio
async def test_different_domains_separate():
    robots_txt = "User-agent: *\nAllow: /"
    resp = _mock_response(200, robots_txt)
    with _patch_httpx(resp) as mock_cls:
        policy = RobotsPolicy(user_agent="TestBot")
        await policy.is_allowed("https://a.com/page")
        await policy.is_allowed("https://b.com/page")
        client = mock_cls.return_value
        assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_500_blocks():
    with _patch_httpx(_mock_response(500)):
        policy = RobotsPolicy(user_agent="TestBot")
        assert await policy.is_allowed("https://example.com/page") is False


@pytest.mark.asyncio
async def test_clear_cache():
    robots_txt = "User-agent: *\nAllow: /"
    with _patch_httpx(_mock_response(200, robots_txt)):
        policy = RobotsPolicy(user_agent="TestBot")
        await policy.is_allowed("https://example.com/page")
        policy.clear_cache()
        await policy.is_allowed("https://example.com/page")
