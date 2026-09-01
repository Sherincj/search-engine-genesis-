"""
Phase 1 API tests.

Verifies:
- /health returns 200 with expected shape
- Placeholder endpoints return 501
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    """Async test client that exercises the full ASGI lifespan."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Health endpoint ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_shape(client):
    data = (await client.get("/health")).json()
    assert data["status"] == "ok"
    assert "index_loaded" in data
    assert "document_count" in data


# ── Placeholder endpoints return 501 ────────────────────────

@pytest.mark.asyncio
async def test_search_not_implemented(client):
    response = await client.get("/search")
    assert response.status_code == 501


@pytest.mark.asyncio
async def test_documents_post_not_implemented(client):
    response = await client.post("/documents")
    assert response.status_code == 501


@pytest.mark.asyncio
async def test_documents_get_not_implemented(client):
    response = await client.get("/documents/1")
    assert response.status_code == 501


@pytest.mark.asyncio
async def test_documents_delete_not_implemented(client):
    response = await client.delete("/documents/1")
    assert response.status_code == 501


@pytest.mark.asyncio
async def test_crawl_post_not_implemented(client):
    response = await client.post("/crawl")
    assert response.status_code == 501


@pytest.mark.asyncio
async def test_crawl_get_not_implemented(client):
    response = await client.get("/crawl/some-job-id")
    assert response.status_code == 501
