# Search Engine — README

A search engine built from scratch for learning. Implements tokenization,
inverted indexing, TF-IDF, BM25, crawling, and snippet generation — all
without third-party search libraries.

## Architecture

```
┌──────────────────────────┐
│       NEXT.JS UI         │
│  Search Box / Results    │
└────────────┬─────────────┘
             │ HTTP / JSON
             ▼
┌──────────────────────────┐
│       FASTAPI            │
│  /health /search         │
│  /documents /crawl       │
└────────────┬─────────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
 Crawler  Search   Document
          Engine   Storage
    │        │     (SQLite)
    ▼        ▼
 Parser   Index
          (inverted)
```

## Tech Stack

| Layer    | Technology                        |
|----------|-----------------------------------|
| Backend  | Python 3.12+, FastAPI, SQLAlchemy |
| Database | SQLite (async via aiosqlite)      |
| Frontend | Next.js, React, TypeScript        |
| Testing  | pytest, pytest-asyncio            |

## Project Structure

```
search-engine/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routers
│   │   ├── config/        # Pydantic settings
│   │   ├── crawler/       # Web crawler (Phase 10)
│   │   ├── database/      # SQLAlchemy engine, models & repositories
│   │   ├── indexing/      # Inverted index (Phase 4)
│   │   ├── models/        # Pydantic schemas
│   │   ├── parser/        # HTML parser (Phase 10)
│   │   ├── search/        # Search engine (Phase 5)
│   │   ├── utils/         # Logging, timing
│   │   └── main.py        # App entry-point
│   ├── tests/
│   ├── pyproject.toml
│   └── requirements.txt
├── frontend/
│   ├── app/               # Next.js pages
│   ├── components/        # React components
│   ├── lib/               # API client & types
│   └── package.json
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- npm

### Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e ".[dev]"
```

### Frontend

```bash
cd frontend
npm install
```

### Environment

Copy the example env file and adjust if needed:

```bash
cp .env.example .env
```

## Running

### Backend

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm run dev
```

The UI will be available at `http://localhost:3000`.

## API Endpoints

| Method | Endpoint          | Status       |
|--------|-------------------|-------------|
| GET    | /health           | ✅ Working   |
| POST   | /documents        | ✅ Working   |
| GET    | /documents/{id}   | ✅ Working   |
| DELETE | /documents/{id}   | ✅ Working   |
| GET    | /search?q=…       | 🚧 Phase 5  |
| POST   | /crawl            | 🚧 Phase 10 |
| GET    | /crawl/{job_id}   | 🚧 Phase 10 |

## Document Storage (Phase 2)

### Document Model

Every stored document has the following fields:

| Field          | Type     | Notes                                      |
|----------------|----------|--------------------------------------------|
| `id`           | Integer  | Auto-incrementing primary key              |
| `url`          | String   | **Unique**, indexed — the source URL       |
| `canonical_url`| String   | Optional — canonical URL if provided       |
| `domain`       | String   | Indexed — derived from URL by the backend  |
| `title`        | String   | Document title (required)                  |
| `text`         | Text     | Full document text (required)              |
| `content_hash` | String   | SHA-256 of text — **indexed but NOT unique**|
| `word_count`   | Integer  | Derived — whitespace-based count           |
| `created_at`   | DateTime | UTC — set on creation                      |
| `updated_at`   | DateTime | UTC — set on creation and every update     |
| `status`       | String   | Defaults to `"stored"`                     |

### Why `content_hash` is indexed but not unique

Different URLs can legitimately contain identical content. For example,
`/page1` and `/page2` may mirror each other. Both documents are allowed
to exist. The hash index enables fast duplicate-content lookups for
future deduplication in the ranking/indexing phases.

### Create a Document

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/python",
    "title": "Python Tutorial",
    "text": "Python is a powerful programming language."
  }'
```

Response (201 Created):
```json
{
  "id": 1,
  "url": "https://example.com/python",
  "canonical_url": null,
  "domain": "example.com",
  "title": "Python Tutorial",
  "text": "Python is a powerful programming language.",
  "word_count": 7,
  "content_hash": "a1b2c3...",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z",
  "status": "stored"
}
```

### Retrieve a Document

```bash
curl http://localhost:8000/documents/1
```

Response (200 OK): same shape as above.

### Delete a Document

```bash
curl -X DELETE http://localhost:8000/documents/1
```

Response: 204 No Content.

### Duplicate URL Handling

Attempting to POST a document with the same URL as an existing document
returns **409 Conflict**:

```json
{
  "detail": "A document with this URL already exists."
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "database": "connected",
  "index_loaded": false,
  "document_count": 1
}
```

The document count is always read from the actual database.

## Testing

```bash
cd backend
pytest tests/ -v
```

## Development Phases

- [x] **Phase 1** — Project structure, FastAPI, Next.js, SQLite, pytest
- [x] **Phase 2** — Document storage & CRUD API
- [ ] **Phase 3** — Tokenizer
- [ ] **Phase 4** — Inverted index
- [ ] **Phase 5** — Simple keyword search
- [ ] **Phase 6** — TF-IDF
- [ ] **Phase 7** — BM25
- [ ] **Phase 8** — Field weighting
- [ ] **Phase 9** — Snippet generation
- [ ] **Phase 10** — Crawler
- [ ] **Phase 11** — Crawler → parser → index pipeline
- [ ] **Phase 12** — Frontend search UI
- [ ] **Phase 13** — Scale testing

## License

MIT