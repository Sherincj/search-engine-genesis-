"""
Search API and end-to-end integration tests.

Uses an in-memory SQLite database (via dependency override) with
a manually built inverted index injected into the search service.

Covers:
- GET /search?q=... returns 200
- Empty query returns 400
- Unknown query returns 200 with zero results
- Multiple terms
- Ranking
- Pagination
- Page boundaries
- Invalid page / limit
- Empty index
- Duplicate query terms
- Missing database document (stale index reference)
- Timing field
- Response shape
- Repository get_by_ids (order preservation + missing IDs)
- Full end-to-end: create → index → persist → load → search → verify
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.database.database import get_db
from app.database.models import Base
from app.database.repositories import DocumentRepository
from app.indexing import DocumentIndexer, IndexStorage, InvertedIndex
from app.main import app
from app.services.search_service import SearchService, set_search_service


# ── In-memory database setup ────────────────────────────────

_engine = create_async_engine("sqlite+aiosqlite://", echo=False)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def _override_get_db():
    async with _session_factory() as session:
        yield session


# ── Test data ────────────────────────────────────────────────

DOCS = [
    {
        "url": "https://example.com/python",
        "title": "Python Tutorial",
        "text": "Python is a programming language.",
    },
    {
        "url": "https://example.com/javascript",
        "title": "JavaScript Tutorial",
        "text": "JavaScript is a programming language for the web.",
    },
    {
        "url": "https://example.com/python-web",
        "title": "Python Web Development",
        "text": "Python can be used for web development.",
    },
    {
        "url": "https://example.com/database",
        "title": "Database Guide",
        "text": "Databases store structured information.",
    },
]


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
async def _setup_db():
    """Create / drop tables for each test and scope the dependency override."""
    app.dependency_overrides[get_db] = _override_get_db
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    # Reset search service after each test
    set_search_service(SearchService(InvertedIndex()))
    # Remove our override so it doesn't leak into other test modules
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def populated_client():
    """Client with documents and a built index."""
    # Insert docs
    async with _session_factory() as session:
        repo = DocumentRepository(session)
        created_docs = []
        for d in DOCS:
            doc = await repo.create_document(
                url=d["url"], title=d["title"], text=d["text"]
            )
            created_docs.append(doc)

    # Build index
    index = InvertedIndex()
    indexer = DocumentIndexer(index)
    for doc in created_docs:
        indexer.index_document(doc.id, doc.text)

    set_search_service(SearchService(index))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Helper ───────────────────────────────────────────────────

async def _create_and_index_docs(session: AsyncSession) -> InvertedIndex:
    """Insert test docs into DB and build an index."""
    repo = DocumentRepository(session)
    index = InvertedIndex()
    indexer = DocumentIndexer(index)

    for d in DOCS:
        doc = await repo.create_document(
            url=d["url"], title=d["title"], text=d["text"]
        )
        indexer.index_document(doc.id, doc.text)

    return index


# ── Basic search ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_returns_200(populated_client):
    resp = await populated_client.get("/search", params={"q": "python"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_search_response_shape(populated_client):
    resp = await populated_client.get("/search", params={"q": "python"})
    data = resp.json()

    assert "query" in data
    assert "total" in data
    assert "page" in data
    assert "limit" in data
    assert "results" in data
    assert "took_ms" in data
    assert data["query"] == "python"


@pytest.mark.asyncio
async def test_search_result_item_shape(populated_client):
    resp = await populated_client.get("/search", params={"q": "python"})
    results = resp.json()["results"]

    assert len(results) > 0
    item = results[0]
    assert "document_id" in item
    assert "title" in item
    assert "url" in item
    assert "score" in item


# ── Empty query → 400 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_query_400(client):
    resp = await client.get("/search", params={"q": ""})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_missing_query_400(client):
    resp = await client.get("/search")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_whitespace_query_400(client):
    resp = await client.get("/search", params={"q": "   "})
    assert resp.status_code == 400


# ── Unknown query → 200 with zero results ────────────────────

@pytest.mark.asyncio
async def test_unknown_query_200(populated_client):
    resp = await populated_client.get("/search", params={"q": "zzznonsense"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["results"] == []


# ── Multiple terms ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_term_search(populated_client):
    resp = await populated_client.get("/search", params={"q": "python web"})
    data = resp.json()

    assert data["total"] > 0
    # "Python Web Development" doc matches both terms
    titles = [r["title"] for r in data["results"]]
    assert "Python Web Development" in titles


# ── Ranking ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ranking_multi_term(populated_client):
    """
    'python web' — Doc "Python Web Development" matches both terms
    and should rank first.
    """
    resp = await populated_client.get("/search", params={"q": "python web"})
    data = resp.json()

    assert data["results"][0]["title"] == "Python Web Development"


# ── Pagination ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pagination_limit(populated_client):
    resp = await populated_client.get(
        "/search", params={"q": "programming", "limit": 1}
    )
    data = resp.json()

    assert len(data["results"]) == 1
    assert data["total"] >= 2  # At least 2 docs mention "programming"
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_pagination_page2(populated_client):
    resp = await populated_client.get(
        "/search", params={"q": "programming", "page": 2, "limit": 1}
    )
    data = resp.json()

    assert data["page"] == 2
    assert len(data["results"]) <= 1


@pytest.mark.asyncio
async def test_pagination_beyond_results(populated_client):
    resp = await populated_client.get(
        "/search", params={"q": "python", "page": 100}
    )
    data = resp.json()

    assert data["results"] == []
    assert data["total"] > 0  # Documents exist, just past last page


# ── Invalid page / limit ────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_page_zero(client):
    resp = await client.get("/search", params={"q": "python", "page": 0})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_limit_zero(client):
    resp = await client.get("/search", params={"q": "python", "limit": 0})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_limit_too_large(client):
    resp = await client.get("/search", params={"q": "python", "limit": 200})
    assert resp.status_code == 422


# ── Empty index ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_empty_index(client):
    """Search on an empty index returns zero results, not an error."""
    set_search_service(SearchService(InvertedIndex()))
    resp = await client.get("/search", params={"q": "python"})

    assert resp.status_code == 200
    assert resp.json()["total"] == 0


# ── Duplicate query terms ────────────────────────────────────

@pytest.mark.asyncio
async def test_duplicate_query_terms(populated_client):
    resp = await populated_client.get(
        "/search", params={"q": "python python python"}
    )
    data = resp.json()

    # Should return same results as a single "python"
    single = await populated_client.get("/search", params={"q": "python"})
    single_data = single.json()

    assert data["total"] == single_data["total"]


# ── Timing field ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timing_field(populated_client):
    resp = await populated_client.get("/search", params={"q": "python"})
    data = resp.json()

    assert "took_ms" in data
    assert isinstance(data["took_ms"], (int, float))
    assert data["took_ms"] >= 0


# ── Repository: get_by_ids ───────────────────────────────────

@pytest.mark.asyncio
async def test_get_by_ids_preserves_order():
    async with _session_factory() as session:
        repo = DocumentRepository(session)
        doc1 = await repo.create_document(
            url="https://a.com/1", title="Doc A", text="alpha"
        )
        doc2 = await repo.create_document(
            url="https://b.com/2", title="Doc B", text="beta"
        )
        doc3 = await repo.create_document(
            url="https://c.com/3", title="Doc C", text="gamma"
        )

        # Request in reverse order
        docs = await repo.get_by_ids([doc3.id, doc1.id, doc2.id])
        assert [d.id for d in docs] == [doc3.id, doc1.id, doc2.id]


@pytest.mark.asyncio
async def test_get_by_ids_skips_missing():
    async with _session_factory() as session:
        repo = DocumentRepository(session)
        doc1 = await repo.create_document(
            url="https://a.com/1", title="Doc A", text="alpha"
        )

        docs = await repo.get_by_ids([doc1.id, 9999])
        assert len(docs) == 1
        assert docs[0].id == doc1.id


@pytest.mark.asyncio
async def test_get_by_ids_empty():
    async with _session_factory() as session:
        repo = DocumentRepository(session)
        docs = await repo.get_by_ids([])
        assert docs == []


# ── Missing database document (stale index reference) ────────

@pytest.mark.asyncio
async def test_stale_index_reference():
    """If the index references a deleted document, skip it gracefully."""
    async with _session_factory() as session:
        repo = DocumentRepository(session)
        doc = await repo.create_document(
            url="https://example.com/stale",
            title="Stale",
            text="python test",
        )

        # Build index with this document
        index = InvertedIndex()
        DocumentIndexer(index).index_document(doc.id, doc.text)

        # Delete the document from DB
        await repo.delete_document(doc.id)

        # Search should not crash — just skip the missing doc
        service = SearchService(index)
        set_search_service(service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/search", params={"q": "python"})
        assert resp.status_code == 200
        # The stale doc should be skipped in results
        data = resp.json()
        assert all(r["document_id"] != doc.id for r in data["results"])


# ── Full end-to-end: create → index → persist → load → search ───

@pytest.mark.asyncio
async def test_end_to_end(tmp_path):
    """Complete integration test covering the full search pipeline."""
    # 1. Create documents in SQLite
    async with _session_factory() as session:
        index = await _create_and_index_docs(session)

    # 2. Persist index
    storage = IndexStorage(tmp_path / "test_index")
    storage.save(index)

    # 3. Load index from disk
    loaded_index = storage.load()
    assert loaded_index.validate() == []

    # 4. Verify loaded index is equivalent
    assert loaded_index.get_total_documents() == index.get_total_documents()
    assert loaded_index.vocabulary == index.vocabulary

    # 5. Execute search via service
    service = SearchService(loaded_index, storage)
    set_search_service(service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Search: python
        resp = await ac.get("/search", params={"q": "python"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        python_titles = {r["title"] for r in data["results"]}
        assert "Python Tutorial" in python_titles
        assert "Python Web Development" in python_titles

        # Search: programming language
        resp2 = await ac.get("/search", params={"q": "programming language"})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["total"] >= 2

        # Search: python web → Python Web Development should rank first
        resp3 = await ac.get("/search", params={"q": "python web"})
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["results"][0]["title"] == "Python Web Development"

        # Search: unknown → 200, zero results
        resp4 = await ac.get("/search", params={"q": "xyznonexistent"})
        assert resp4.status_code == 200
        assert resp4.json()["total"] == 0


# ── Index rebuild ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_index(tmp_path):
    """Verify the index can be rebuilt from SQLite documents."""
    async with _session_factory() as session:
        repo = DocumentRepository(session)
        for d in DOCS:
            await repo.create_document(
                url=d["url"], title=d["title"], text=d["text"]
            )

        storage = IndexStorage(tmp_path / "rebuild_index")
        service = SearchService(InvertedIndex(), storage)

        # Before rebuild — empty index
        assert service.index.get_total_documents() == 0

        # Rebuild
        await service.rebuild_index(session)

        # After rebuild — all docs indexed
        assert service.index.get_total_documents() == len(DOCS)
        assert service.index.validate() == []

        # Index persisted to disk
        assert storage.exists()
