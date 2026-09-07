# Search — Technical Documentation

## Architecture

```
User Query
    ↓
Next.js Frontend
    ↓
GET /search?q=python&page=1&limit=10
    ↓
FastAPI SearchService
    │
    ├── SearchEngine
    │   ├── QueryParser (Phase 3 Tokenizer)
    │   ├── CandidateRetrieval (Phase 4 InvertedIndex)
    │   └── Ranking (baseline keyword scoring)
    │
    └── DocumentRepository
        └── SQLite (batch retrieval by IDs)
    ↓
SearchResponse (JSON)
    ↓
Frontend renders results
```

---

## Query Parsing

The query parser reuses the **Phase 3 tokenizer** — queries go through
the exact same normalisation as documents:

```
"Python WEB-development"  →  ["python", "web", "development"]
"don't stop"              →  ["dont", "stop"]
"café"                    →  ["café"]
""                        →  []
```

This is critical: if documents are tokenised one way and queries another,
search quality degrades.

See [`text-processing.md`](text-processing.md) for the full tokenization spec.

---

## Candidate Retrieval

Uses **OR semantics**: a document matches if it contains **any** query term.

```
Query:   "python javascript"
Result:  All documents containing "python" OR "javascript"
```

### How it works

1. Deduplicate query terms.
2. For each unique term, look up its posting list in the inverted index.
3. Merge posting lists into a set of candidate documents.
4. For each candidate, record which terms matched and the total TF.

### Important: no full-document scan

Search never scans all SQLite documents. It uses the inverted index
for direct O(1) term lookups. Only matching document IDs are fetched
from the database.

---

## Baseline Ranking

Phase 5 uses a simple ranking algorithm. Ranking criteria (in order):

| Priority | Criterion                | Direction  |
|----------|--------------------------|------------|
| 1        | Unique query terms matched | Descending |
| 2        | Total term frequency      | Descending |
| 3        | Document ID               | Ascending  |

### Example

```
Query: "python javascript"

Doc 1: "python python"
  → matched_terms = 1, TF = 2

Doc 2: "python javascript"
  → matched_terms = 2, TF = 2

Doc 3: "javascript javascript javascript"
  → matched_terms = 1, TF = 3

Ranking: Doc 2, Doc 3, Doc 1
```

### Why this is simple

This is a baseline. It does not consider:
- How rare a term is (IDF)
- Document length normalization
- Field weighting

TF-IDF (Phase 6) and BM25 (Phase 7) will replace this scoring.

---

## Database Lookup

After ranking, the search service fetches full document records from
SQLite using `get_by_ids()`:

```
Search ranking:  [42, 7, 13, 1]
Database query:  SELECT * FROM documents WHERE id IN (42, 7, 13, 1)
Database result: [1, 7, 13, 42]   ← different order!
Final result:    [42, 7, 13, 1]   ← search order preserved
```

If the index references a document that no longer exists in SQLite
(stale reference), that document is silently skipped.

---

## Pagination

Pagination is applied after ranking:

```
total = 25, limit = 10

Page 1 → results 1–10
Page 2 → results 11–20
Page 3 → results 21–25
```

API parameters:
- `page` — 1-based page number (default: 1, minimum: 1)
- `limit` — results per page (default: 10, range: 1–100)

---

## Index Loading

At service initialization:

```
IndexStorage.exists()?
    ├── Yes → load from disk
    └── No  → start with empty index
```

If the index file is corrupt, log an error and start with an empty index.
The application never crashes on startup due to index issues.

---

## Index Rebuild

The search service can rebuild the entire index from SQLite:

```
SQLite documents → Tokenizer → InvertedIndex → IndexStorage.save()
```

This is useful when the index is deleted or becomes stale.

---

## Timing

The API response includes `took_ms` — total search time in milliseconds:

```json
{
  "took_ms": 1.42
}
```

---

## API Reference

### `GET /search`

| Parameter | Type | Default | Validation         |
|-----------|------|---------|--------------------|
| `q`       | str  | —       | Required, non-empty |
| `page`    | int  | 1       | ≥ 1                |
| `limit`   | int  | 10      | 1–100              |

### Responses

**200 OK** — Results found (or zero results for unknown terms):

```json
{
  "query": "python programming",
  "total": 3,
  "page": 1,
  "limit": 10,
  "results": [
    {
      "document_id": 5,
      "title": "Python Programming Guide",
      "url": "https://example.com/python",
      "score": 2.0
    }
  ],
  "took_ms": 1.42
}
```

**400 Bad Request** — Empty or missing query:

```json
{
  "detail": "Query parameter 'q' is required."
}
```

**422 Unprocessable Entity** — Invalid page or limit.

---

## Current Limitations

Phase 5 does **not** implement:

- **TF-IDF** — term rarity is not considered (Phase 6)
- **BM25** — no document length normalization (Phase 7)
- **Semantic search** — no embeddings or vectors
- **Phrase search** — "python web" matches documents with either term
- **Snippet generation** — no highlighted excerpts
- **Crawler** — no URL fetching
- **Spelling correction** — exact matches only
- **Field weighting** — title and text are not weighted differently
