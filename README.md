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
│   │   ├── text_processing/ # Tokeniser, normaliser, stop words
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

## Text Processing (Phase 3)

The tokeniser converts raw text into normalised tokens used for word counts,
term frequency, and (in future phases) the inverted index, TF-IDF, and BM25.

### Why tokenisation exists

Raw text can't be compared reliably — `"Python"`, `"PYTHON"`, and `"python"`
must all match. The tokeniser normalises text so documents and queries produce
comparable tokens.

### Pipeline

```
Raw text → Unicode NFC → Lowercase → Apostrophe strip → Token extraction → [Optional stop-word filter] → Tokens
```

### Default behaviour

```python
tokenize("Python is GREAT!")          # ['python', 'is', 'great']
tokenize("web-development")          # ['web', 'development']
tokenize("don't stop")               # ['dont', 'stop']
tokenize("python3 web3")             # ['python3', 'web3']
tokenize("café naïve 中文")           # ['café', 'naïve', '中文']
tokenize("Python Python Python web") # ['python', 'python', 'python', 'web']
```

### Key rules

| Input              | Tokens                    | Rule                    |
|--------------------|---------------------------|-------------------------|
| `web-development`  | `["web", "development"]`  | Hyphens split           |
| `don't`            | `["dont"]`                | Apostrophes stripped    |
| `3.14`             | `["3", "14"]`             | Dots split              |
| `python3`          | `["python3"]`             | Alphanumeric kept whole |
| `Python PYTHON`    | `["python", "python"]`    | Lowercase normalised    |

### Stop words

Disabled by default to preserve queries like `"to be or not to be"`:

```python
tokenize("python is a language")                          # ['python', 'is', 'a', 'language']
tokenize("python is a language", remove_stopwords=True)   # ['python', 'language']
```

### Word count

Documents use the tokeniser for `word_count` (not `text.split()`):

```
"hello,world! foo-bar"  →  tokenize  →  ["hello", "world", "foo", "bar"]  →  word_count: 4
```

### No stemming

Stemming is **not implemented** — `programming` stays `programming`, not `program`.
Can be evaluated later based on search quality.

### Technical docs

See [`docs/text-processing.md`](docs/text-processing.md) for the full specification.

## Inverted Index (Phase 4)

The inverted index enables fast term lookups by reversing the document→term
relationship:

```
Without index:  search("python") → scan every document
With index:     search("python") → [doc 1, doc 4, doc 17]  (direct lookup)
```

### Architecture

```
Document → Tokenizer → Tokens → InvertedIndex → IndexStorage (JSON)
```

### What the index stores

| Data                  | Purpose                  | Future consumer |
|-----------------------|--------------------------|-----------------|
| Term → posting list   | Which docs contain term  | Search          |
| Term frequency (TF)   | How often in each doc    | TF-IDF, BM25    |
| Token positions       | Where in the document    | Phrase search   |
| Document frequency    | How many docs have term  | TF-IDF, BM25    |
| Document lengths      | Token count per doc      | BM25            |
| Average doc length    | Mean across all docs     | BM25            |

### Example

```
Document 1: "Python web Python"
Document 2: "JavaScript web"

Index:
  python     → doc 1 (TF=2, pos=[0,2])
  javascript → doc 2 (TF=1, pos=[0])
  web        → doc 1 (TF=1, pos=[1]), doc 2 (TF=1, pos=[1])
```

### Key design rules

- **Tokens come from the Phase 3 tokenizer** — no duplicate normalization
- **SQLite is the source of truth** — the index is derived data
- **Duplicate document re-indexing replaces old postings** — no corruption
- **Empty posting lists are cleaned up automatically**
- **Posting lists are ordered by document ID**

### Persistence

The index is saved as JSON with atomic writes (temp file → rename).
See [`docs/inverted-index.md`](docs/inverted-index.md) for the full specification.

## Basic Keyword Search (Phase 5)

The first real search pipeline:

```
GET /search?q=python+web&page=1&limit=10
```

### How it works

```
User query → Tokenizer → Inverted Index → Ranking → SQLite → Response
```

1. **Query parsing** — same tokenizer as documents (consistent normalisation)
2. **Candidate retrieval** — OR semantics via inverted index posting lists
3. **Ranking** — unique matched terms > total TF > document ID
4. **Database lookup** — batch fetch, search order preserved
5. **Pagination** — page/limit with deterministic ordering

### Example response

```json
{
  "query": "python web",
  "total": 3,
  "page": 1,
  "limit": 10,
  "results": [
    {
      "document_id": 3,
      "title": "Python Web Development",
      "url": "https://example.com/python-web",
      "score": 0.8241
    }
  ],
  "took_ms": 1.42
}
```

### Frontend

A Next.js search UI is available at `http://localhost:3000` with:
- Search input + button
- Result list (title, URL, score)
- Pagination

See [`docs/search.md`](docs/search.md) for the full search specification.

## Ranking (Phase 6 → Phase 7)

### BM25 (Default — Phase 7)

Search results are ranked using **BM25**:

```
BM25(D,Q) = Σ IDF(t) × f(t,D)×(k1+1) / (f(t,D) + k1×(1 - b + b×|D|/avgdl))
IDF(t)    = log(1 + (N - DF + 0.5) / (DF + 0.5))
```

Default parameters:

```
BM25_K1 = 1.2    — term-frequency saturation
BM25_B  = 0.75   — document-length normalization
```

Key behaviors:
- **Rare terms** receive higher IDF → more influence on ranking
- **TF saturation** via k1 — diminishing returns for repeated terms
- **Document length normalization** via b — shorter docs rewarded
- **Scores** are floats with no fixed upper bound (can exceed 1.0)
- **Tie-breaker**: document ID ascending

### TF-IDF (Phase 6, preserved)

The Phase 6 TF-IDF cosine similarity implementation is retained for
comparison and experimentation. See [`docs/tf-idf.md`](docs/tf-idf.md).

### Field Weighting (Phase 8)

The search engine distinguishes between document fields (title vs body) and ranks them independently:

```
Score = (2.0 × Title BM25) + (1.0 × Body BM25)
```

See [`docs/field-weighting.md`](docs/field-weighting.md) for the full specification.

### Not yet implemented

- Snippet generation (Phase 9)
- Crawler (Phase 10)

See [`docs/bm25.md`](docs/bm25.md) for the full BM25 specification.

## Testing

```bash
cd backend
pytest tests/ -v
```

## Development Phases

- [x] **Phase 1** — Project structure, FastAPI, Next.js, SQLite, pytest
- [x] **Phase 2** — Document storage & CRUD API
- [x] **Phase 3** — Tokenizer
- [x] **Phase 4** — Inverted index
- [x] **Phase 5** — Basic keyword search
- [x] **Phase 6** — TF-IDF ranking
- [x] **Phase 7** — BM25 ranking
- [x] **Phase 8** — Field weighting
- [ ] **Phase 9** — Snippet generation
- [ ] **Phase 10** — Crawler
- [ ] **Phase 11** — Crawler → parser → index pipeline
- [ ] **Phase 12** — Frontend search UI
- [ ] **Phase 13** — Scale testing

## License

MIT