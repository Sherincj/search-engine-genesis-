"""
Phase 2 — Document storage tests.

Uses an isolated in-memory SQLite database so no test data
pollutes the development database.

Test matrix:
 1. Create document — 201 + all fields
 2. Content hash determinism
 3. Duplicate URL — 409
 4. Same content, different URLs — both succeed
 5. Get document — 200 + field verification
 6. Get nonexistent document — 404
 7. Delete document — 204 → 404 on re-fetch
 8. Delete nonexistent document — 404
 9. Health document count reflects reality
10. Invalid input (missing fields, bad URL, empty text)
"""

import hashlib

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.database.models import Base
from app.database.database import get_db
from app.main import app


# ── Isolated in-memory database ──────────────────────────────

_test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=False,
    future=True,
)
_test_session_factory = async_sessionmaker(
    bind=_test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def _override_get_db():
    """Yield a session from the in-memory test database."""
    async with _test_session_factory() as session:
        yield session


# Override the production get_db with our test version.
app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(autouse=True)
async def _setup_db():
    """Create all tables before each test and drop them after."""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client():
    """Async test client bound to the app (no lifespan — we manage DB ourselves)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Helpers ──────────────────────────────────────────────────

SAMPLE_DOC = {
    "url": "https://example.com/python",
    "title": "Python Tutorial",
    "text": "Python is a powerful programming language used for web development.",
}


async def _create_doc(client: AsyncClient, **overrides) -> dict:
    """POST a document and return the JSON response body."""
    payload = {**SAMPLE_DOC, **overrides}
    resp = await client.post("/documents", json=payload)
    return resp


# ── Test 1 — Create Document ────────────────────────────────

@pytest.mark.asyncio
async def test_create_document(client):
    resp = await _create_doc(client)
    assert resp.status_code == 201

    body = resp.json()
    assert body["id"] is not None
    assert body["url"] == SAMPLE_DOC["url"]
    assert body["title"] == SAMPLE_DOC["title"]
    assert body["text"] == SAMPLE_DOC["text"]
    assert body["domain"] == "example.com"
    assert body["word_count"] == len(SAMPLE_DOC["text"].split())
    assert body["content_hash"] == hashlib.sha256(
        SAMPLE_DOC["text"].encode("utf-8")
    ).hexdigest()
    assert body["status"] == "stored"
    assert body["created_at"] is not None
    assert body["updated_at"] is not None
    assert body["canonical_url"] is None


@pytest.mark.asyncio
async def test_create_document_with_canonical_url(client):
    resp = await _create_doc(
        client, canonical_url="https://example.com/python-canonical"
    )
    assert resp.status_code == 201
    assert resp.json()["canonical_url"] == "https://example.com/python-canonical"


# ── Test 2 — Content Hash Determinism ────────────────────────

@pytest.mark.asyncio
async def test_content_hash_determinism(client):
    """Same text must always produce the same SHA-256 hash."""
    text = "Deterministic hash test content."
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()

    resp = await _create_doc(
        client,
        url="https://example.com/hash-test",
        text=text,
    )
    assert resp.json()["content_hash"] == expected


# ── Test 3 — Duplicate URL → 409 ────────────────────────────

@pytest.mark.asyncio
async def test_create_duplicate_document(client):
    resp1 = await _create_doc(client)
    assert resp1.status_code == 201

    resp2 = await _create_doc(client)  # same URL
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"].lower()


# ── Test 4 — Same Content, Different URLs ────────────────────

@pytest.mark.asyncio
async def test_same_content_different_urls(client):
    shared_text = "Hello world"
    resp1 = await _create_doc(
        client,
        url="https://example.com/page1",
        text=shared_text,
    )
    resp2 = await _create_doc(
        client,
        url="https://example.com/page2",
        text=shared_text,
    )
    assert resp1.status_code == 201
    assert resp2.status_code == 201

    # Both have the same content hash
    assert resp1.json()["content_hash"] == resp2.json()["content_hash"]
    # But different IDs
    assert resp1.json()["id"] != resp2.json()["id"]


# ── Test 5 — Get Document ───────────────────────────────────

@pytest.mark.asyncio
async def test_get_document(client):
    create_resp = await _create_doc(client)
    doc_id = create_resp.json()["id"]

    get_resp = await client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 200

    body = get_resp.json()
    assert body["id"] == doc_id
    assert body["url"] == SAMPLE_DOC["url"]
    assert body["title"] == SAMPLE_DOC["title"]
    assert body["text"] == SAMPLE_DOC["text"]
    assert body["domain"] == "example.com"
    assert body["status"] == "stored"


# ── Test 6 — Get Nonexistent Document → 404 ──────────────────

@pytest.mark.asyncio
async def test_get_nonexistent_document(client):
    resp = await client.get("/documents/999999")
    assert resp.status_code == 404


# ── Test 7 — Delete Document ────────────────────────────────

@pytest.mark.asyncio
async def test_delete_document(client):
    create_resp = await _create_doc(client)
    doc_id = create_resp.json()["id"]

    # Delete succeeds
    del_resp = await client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 204

    # Re-fetch returns 404
    get_resp = await client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 404


# ── Test 8 — Delete Nonexistent Document → 404 ───────────────

@pytest.mark.asyncio
async def test_delete_nonexistent_document(client):
    resp = await client.delete("/documents/999999")
    assert resp.status_code == 404


# ── Test 9 — Health Document Count ───────────────────────────

@pytest.mark.asyncio
async def test_health_document_count(client):
    # Initially 0
    health = (await client.get("/health")).json()
    assert health["document_count"] == 0

    # Create one
    await _create_doc(client, url="https://example.com/a")
    health = (await client.get("/health")).json()
    assert health["document_count"] == 1

    # Create another
    resp2 = await _create_doc(client, url="https://example.com/b")
    doc2_id = resp2.json()["id"]
    health = (await client.get("/health")).json()
    assert health["document_count"] == 2

    # Delete one
    await client.delete(f"/documents/{doc2_id}")
    health = (await client.get("/health")).json()
    assert health["document_count"] == 1


# ── Test 10 — Invalid Input ─────────────────────────────────

@pytest.mark.asyncio
async def test_missing_url(client):
    resp = await client.post("/documents", json={
        "title": "Title",
        "text": "Some text",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_missing_title(client):
    resp = await client.post("/documents", json={
        "url": "https://example.com/test",
        "text": "Some text",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_missing_text(client):
    resp = await client.post("/documents", json={
        "url": "https://example.com/test",
        "title": "Title",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_url_scheme(client):
    resp = await client.post("/documents", json={
        "url": "ftp://example.com/test",
        "title": "Title",
        "text": "Some text",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_url_no_scheme(client):
    resp = await client.post("/documents", json={
        "url": "not-a-url",
        "title": "Title",
        "text": "Some text",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_empty_title(client):
    resp = await client.post("/documents", json={
        "url": "https://example.com/test",
        "title": "   ",
        "text": "Some text",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_empty_text(client):
    resp = await client.post("/documents", json={
        "url": "https://example.com/test",
        "title": "Title",
        "text": "   ",
    })
    assert resp.status_code == 422


# ── Test 11 — Domain extraction edge cases ───────────────────

@pytest.mark.asyncio
async def test_domain_strips_www(client):
    """www.example.com → example.com"""
    resp = await _create_doc(
        client,
        url="https://www.example.com/page",
    )
    assert resp.status_code == 201
    assert resp.json()["domain"] == "example.com"


@pytest.mark.asyncio
async def test_domain_preserves_subdomain(client):
    """blog.example.com stays blog.example.com (not stripped)."""
    resp = await _create_doc(
        client,
        url="https://blog.example.com/article",
    )
    assert resp.status_code == 201
    assert resp.json()["domain"] == "blog.example.com"
