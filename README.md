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
│   │   ├── database/      # SQLAlchemy engine & models
│   │   ├── indexing/      # Inverted index (Phase 4)
│   │   ├── models/        # Pydantic schemas (Phase 2)
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
| GET    | /search?q=…       | 🚧 Phase 5  |
| POST   | /documents        | 🚧 Phase 2  |
| GET    | /documents/{id}   | 🚧 Phase 2  |
| DELETE | /documents/{id}   | 🚧 Phase 2  |
| POST   | /crawl            | 🚧 Phase 10 |
| GET    | /crawl/{job_id}   | 🚧 Phase 10 |

## Testing

```bash
cd backend
pytest tests/ -v
```

## Development Phases

- [x] **Phase 1** — Project structure, FastAPI, Next.js, SQLite, pytest
- [ ] **Phase 2** — Document storage & CRUD API
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