# SEARCH ENGINE — FULL IMPLEMENTATION SPECIFICATION

## 1. Project Goal

Build a search engine from scratch using:

* **Python** for the backend, crawler, parser, indexing, ranking, and search engine
* **TypeScript + Next.js + React** for the frontend
* **FastAPI** for communication between frontend and backend
* **SQLite initially** for persistent metadata/document storage
* A custom **inverted index** for the first search implementation
* **BM25** as the primary ranking algorithm after the basic search implementation works
* A modular architecture that can later support:

  * PostgreSQL
  * Redis
  * Elasticsearch/OpenSearch
  * vector/semantic search
  * embeddings
  * hybrid search
  * PageRank/link-based authority
  * distributed crawling/indexing

The first objective is NOT to build a Google-scale search engine.

The first objective is to build a technically correct, understandable search engine that can:

1. Accept documents.
2. Crawl permitted websites.
3. Parse HTML.
4. Extract useful text and metadata.
5. Build an inverted index.
6. Search the index.
7. Rank results.
8. Generate useful snippets.
9. Expose the search functionality through an API.
10. Display results through a Next.js frontend.

The architecture must be designed so the system can evolve from a small local search engine into a much larger search platform.

---

# 2. IMPORTANT DEVELOPMENT PHILOSOPHY

This project is also intended as a programming-learning project.

DO NOT hide the fundamental search-engine implementation behind third-party search libraries.

The following must initially be implemented manually:

* tokenization
* text normalization
* inverted index
* term frequencies
* document frequencies
* TF-IDF
* BM25
* basic query processing
* result scoring
* result ranking
* snippet generation
* URL normalization
* duplicate URL detection
* crawler queue
* crawl state management

Libraries may be used for:

* HTTP requests
* HTML parsing
* database access
* API framework
* frontend
* testing

But do NOT replace the core search-engine algorithms with Elasticsearch, Whoosh, Tantivy, Lucene, or another search engine during V1.

The purpose is to understand how a search engine actually works.

---

# 3. HIGH-LEVEL ARCHITECTURE

The system should follow this architecture:

```
                     ┌──────────────────────────┐
                     │       NEXT.JS UI         │
                     │                          │
                     │ Search Box               │
                     │ Search Results            │
                     │ Pagination                │
                     │ Result Metadata           │
                     └────────────┬─────────────┘
                                  │
                                  │ HTTP/JSON
                                  ▼
                     ┌──────────────────────────┐
                     │       FASTAPI            │
                     │                          │
                     │ /search                  │
                     │ /documents               │
                     │ /crawl                   │
                     │ /health                  │
                     └────────────┬─────────────┘
                                  │
                ┌─────────────────┼──────────────────┐
                │                 │                  │
                ▼                 ▼                  ▼
         ┌────────────┐   ┌──────────────┐   ┌─────────────┐
         │  CRAWLER   │   │ SEARCH       │   │ DOCUMENT    │
         │            │   │ ENGINE       │   │ STORAGE     │
         │ URL Queue  │   │              │   │             │
         │ Fetcher    │   │ Query Parser │   │ SQLite      │
         │ Robots     │   │ Index        │   │ Metadata    │
         └─────┬──────┘   │ Ranking      │   └─────────────┘
               │          └──────┬───────┘
               ▼                 │
         ┌────────────┐          │
         │ HTML       │          │
         │ PARSER     │          │
         │            │          │
         │ Text       │          │
         │ Title      │          │
         │ Links      │          │
         │ Metadata   │          │
         └─────┬──────┘          │
               │                 │
               └────────┬────────┘
                        ▼
                ┌──────────────┐
                │ SEARCH INDEX │
                │              │
                │ term -> docs │
                │ TF           │
                │ DF           │
                │ positions    │
                └──────────────┘
```

# 4. TECHNOLOGY STACK

## Backend

Use:

* Python 3.12+
* FastAPI
* Uvicorn
* Pydantic
* httpx
* BeautifulSoup4
* SQLite
* SQLAlchemy
* pytest

Optional later:

* aiohttp
* Redis
* PostgreSQL
* Qdrant/pgvector
* NumPy

Do not introduce optional technologies until the architecture actually requires them.

---

# 5. FRONTEND STACK

Use:

* Next.js
* React
* TypeScript
* CSS/Tailwind only if useful

Frontend responsibilities:

* search input
* query submission
* loading state
* error state
* results rendering
* pagination
* result title
* URL
* snippet
* search timing
* number of results

The frontend should NOT contain search-engine logic.

The frontend communicates with the Python API.

---

# 6. PROJECT STRUCTURE

Create the following structure:

```text
search-engine/
│
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
│
├── backend/
│   ├── requirements.txt
│   ├── pyproject.toml
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   │
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── search.py
│   │   │   ├── crawl.py
│   │   │   ├── documents.py
│   │   │   └── health.py
│   │   │
│   │   ├── crawler/
│   │   │   ├── __init__.py
│   │   │   ├── crawler.py
│   │   │   ├── fetcher.py
│   │   │   ├── url_queue.py
│   │   │   ├── robots.py
│   │   │   ├── url_normalizer.py
│   │   │   └── crawl_state.py
│   │   │
│   │   ├── parser/
│   │   │   ├── __init__.py
│   │   │   ├── html_parser.py
│   │   │   ├── text_extractor.py
│   │   │   ├── metadata_extractor.py
│   │   │   └── link_extractor.py
│   │   │
│   │   ├── indexing/
│   │   │   ├── __init__.py
│   │   │   ├── inverted_index.py
│   │   │   ├── index_builder.py
│   │   │   ├── index_storage.py
│   │   │   └── document_stats.py
│   │   │
│   │   ├── search/
│   │   │   ├── __init__.py
│   │   │   ├── query_parser.py
│   │   │   ├── search_engine.py
│   │   │   ├── candidate_retrieval.py
│   │   │   ├── ranking.py
│   │   │   ├── bm25.py
│   │   │   ├── tfidf.py
│   │   │   └── snippet_generator.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── document.py
│   │   │   ├── crawl.py
│   │   │   └── search.py
│   │   │
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   ├── models.py
│   │   │   └── repositories.py
│   │   │
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging.py
│   │       └── timing.py
│   │
│   └── tests/
│       ├── test_crawler.py
│       ├── test_parser.py
│       ├── test_url_normalizer.py
│       ├── test_tokenizer.py
│       ├── test_inverted_index.py
│       ├── test_tfidf.py
│       ├── test_bm25.py
│       ├── test_search.py
│       └── test_api.py
│
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── next.config.ts
    │
    ├── app/
    │   ├── page.tsx
    │   ├── layout.tsx
    │   ├── globals.css
    │   └── search/
    │       └── page.tsx
    │
    ├── components/
    │   ├── SearchBar.tsx
    │   ├── SearchResults.tsx
    │   ├── SearchResult.tsx
    │   ├── Pagination.tsx
    │   ├── LoadingState.tsx
    │   └── ErrorState.tsx
    │
    ├── lib/
    │   ├── api.ts
    │   └── types.ts
    │
    └── public/
```

Keep responsibilities separated.

Do not create a giant `main.py`.

Do not create a giant `search.py`.

Do not put crawler logic into API routes.

Do not put database queries into React components.

---

# 7. DOCUMENT MODEL

Every indexed document should have at least:

```text
Document
├── id
├── url
├── canonical_url
├── domain
├── title
├── text
├── word_count
├── content_hash
├── crawled_at
├── updated_at
└── status
```

The parser should additionally be capable of extracting:

```text
meta description
headings
links
language
canonical URL
```

Later these can become ranking signals.

---

# 8. DATABASE DESIGN

Initially use SQLite.

Create tables conceptually equivalent to:

## documents

```text
id
url
canonical_url
domain
title
text
content_hash
word_count
created_at
updated_at
```

## crawled_urls

```text
id
url
normalized_url
domain
status
depth
discovered_at
crawled_at
last_error
```

## links

```text
id
source_document_id
target_url
anchor_text
```

## crawl_jobs

```text
id
seed_url
status
max_pages
max_depth
started_at
completed_at
pages_crawled
pages_failed
```

Do not store the entire inverted index in a normal relational table initially if an in-memory index is sufficient.

Create an abstraction so index storage can later be replaced.

---

# 9. URL NORMALIZATION

Implement a dedicated URL normalization system.

These URLs should be recognized as potentially equivalent:

```text
https://example.com
https://example.com/
HTTP://EXAMPLE.COM
https://example.com/#section
```

Normalize:

* scheme
* hostname
* default ports
* fragments
* trailing slash where appropriate
* URL encoding
* query parameters according to defined rules

Do NOT blindly remove query parameters because some websites use them to identify content.

The normalization rules must be documented.

---

# 10. CRAWLER

Create a crawler capable of:

1. Receiving seed URLs.
2. Fetching pages.
3. Checking robots.txt.
4. Respecting crawl delays where applicable.
5. Parsing HTML.
6. Extracting links.
7. Normalizing URLs.
8. Avoiding duplicate URLs.
9. Tracking crawl depth.
10. Sending documents to the indexer.
11. Recording failures.

Example:

```text
Seed URL
   ↓
URL Queue
   ↓
robots.txt check
   ↓
HTTP request
   ↓
Response validation
   ↓
HTML parser
   ↓
Extract document
   ↓
Extract links
   ↓
Normalize links
   ↓
Deduplicate
   ↓
Queue new URLs
   ↓
Index document
```

---

# 11. CRAWLER SAFETY REQUIREMENTS

The crawler must:

* identify itself with a configurable User-Agent
* use request timeouts
* implement retry limits
* implement rate limiting
* avoid infinite crawling
* obey robots.txt
* limit maximum pages
* limit crawl depth
* reject unsupported content types
* avoid excessively large responses
* gracefully handle HTTP errors
* gracefully handle malformed HTML
* prevent crawler loops

Default development settings should be conservative.

For example:

```text
max_pages = 100
max_depth = 2
request_timeout = 10 seconds
delay_between_requests = configurable
```

These values must be configurable.

---

# 12. HTML PARSER

The parser should extract:

```text
title
main text
headings
meta description
canonical URL
links
```

It should remove or ignore:

```text
script
style
noscript
navigation where appropriate
```

Do not assume the `<body>` text is always the main content.

Implement a clean abstraction so a better content extraction algorithm can be added later.

---

# 13. TEXT PROCESSING

Create a tokenizer.

Example:

```text
Input:

"Python is an amazing programming language!"

Tokens:

python
is
an
amazing
programming
language
```

Implement:

* lowercase normalization
* punctuation handling
* whitespace normalization
* Unicode handling
* token boundaries

Initially do not over-engineer stemming or lemmatization.

Create the architecture so they can be added later.

---

# 14. STOP WORDS

Support optional stop-word removal.

Example:

```text
the
a
an
is
of
to
in
```

However, make this configurable.

Do not blindly remove every common word because some search queries depend on them.

The tokenizer should be capable of operating in:

```text
raw mode
normalized mode
stopword-filtered mode
```

---

# 15. INVERTED INDEX

Implement a custom inverted index.

Conceptually:

```text
term
  ↓
posting list
```

Example:

```text
python:
    document_1
    document_7
    document_19

javascript:
    document_2
    document_7
    document_22
```

A posting should eventually contain:

```text
document_id
term_frequency
positions
```

Example:

```text
python:
[
    {
        document_id: 1,
        term_frequency: 4,
        positions: [3, 18, 29, 51]
    }
]
```

Positions will later support:

* phrase search
* proximity ranking
* better snippets

---

# 16. DOCUMENT STATISTICS

Maintain:

```text
total_documents
document_frequency(term)
document_length(document)
average_document_length
term_frequency(term, document)
```

These statistics are required for ranking algorithms.

---

# 17. FIRST SEARCH IMPLEMENTATION

Before implementing BM25, implement simple keyword search.

Query:

```text
python web framework
```

Steps:

```text
Query
 ↓
Normalize
 ↓
Tokenize
 ↓
Look up each token
 ↓
Collect candidate documents
 ↓
Calculate simple score
 ↓
Sort
 ↓
Return results
```

Basic score:

```text
score(document) =
    number of matching query terms
```

Then improve it using term frequency.

---

# 18. TF-IDF

Implement TF-IDF manually.

TF:

```text
TF(term, document)
```

IDF:

```text
IDF(term) = log(N / DF(term))
```

TF-IDF:

```text
TF-IDF = TF × IDF
```

Document the mathematical reasoning in comments and documentation.

Do not simply copy a library implementation.

---

# 19. BM25

Implement BM25 manually.

Use the standard formulation:

```text
score(D,Q) =
Σ IDF(qi) *
    [ TF(qi,D) * (k1 + 1) ]
    ---------------------------------------
    [ TF(qi,D) + k1 * (1 - b + b * |D|/avgdl) ]
```

Default parameters:

```text
k1 = 1.2
b = 0.75
```

Make them configurable.

Explain in comments:

* what `k1` does
* what `b` does
* what document length normalization means
* why rare words should matter more

---

# 20. FIELD WEIGHTING

Do not treat all text equally.

Create separate scoring signals for:

```text
title
headings
body
URL
anchor text
meta description
```

Example conceptual ranking:

```text
final_score =
    BM25(body)
    + title_boost
    + heading_boost
    + URL_boost
    + anchor_boost
```

Make the weights configurable.

Do not hard-code them throughout the project.

---

# 21. QUERY PROCESSING

Create a query parser.

Initially support:

```text
normal query
```

Then architect for:

```text
"exact phrase"
word1 AND word2
word1 OR word2
-word
site:example.com
```

Do NOT implement every advanced operator immediately.

Build a clean query representation such as:

```text
ParsedQuery
├── terms
├── phrases
├── excluded_terms
├── site_filter
└── operators
```

---

# 22. CANDIDATE RETRIEVAL

Search should not score every document.

Instead:

```text
Query
 ↓
Term lookup
 ↓
Posting lists
 ↓
Candidate documents
 ↓
Ranking
 ↓
Top K
```

Implement a configurable `top_k`.

For example:

```text
top_k = 20
```

---

# 23. RESULT SNIPPETS

Generate a useful snippet from the document.

For:

```text
python web framework
```

the result might display:

```text
Python is a powerful programming language used
for building modern web applications and APIs...
```

Prefer text surrounding query terms.

Highlight matching terms in the frontend.

Do not generate snippets using an LLM initially.

---

# 24. SEARCH API

Create:

```http
GET /search?q=<query>
```

Support parameters:

```text
q
page
limit
```

Example:

```text
/search?q=python&page=1&limit=10
```

Response:

```json
{
    "query": "python",
    "total": 123,
    "page": 1,
    "limit": 10,
    "took_ms": 4.21,
    "results": [
        {
            "title": "Python Documentation",
            "url": "https://example.com",
            "snippet": "Python is...",
            "score": 12.53
        }
    ]
}
```

Do not expose unnecessary internal implementation details in the public API.

---

# 25. DOCUMENT API

Create APIs for development/admin purposes.

Example:

```http
POST /documents
GET /documents/{id}
DELETE /documents/{id}
```

Allow manually adding documents so the search engine can be tested without crawling.

---

# 26. CRAWL API

Create:

```http
POST /crawl
```

Example request:

```json
{
    "seed_url": "https://example.com",
    "max_pages": 100,
    "max_depth": 2
}
```

Return a crawl job ID.

Example:

```json
{
    "job_id": "abc123",
    "status": "queued"
}
```

Create:

```http
GET /crawl/{job_id}
```

to retrieve status.

---

# 27. HEALTH API

Create:

```http
GET /health
```

Response:

```json
{
    "status": "ok",
    "index_loaded": true,
    "document_count": 1000
}
```

---

# 28. FRONTEND

Build a clean search-engine-style UI.

Home page:

```text
                 SEARCH ENGINE

        ┌────────────────────────────┐
        │ Search the web...       🔍 │
        └────────────────────────────┘
```

After searching:

```text
SEARCH ENGINE

┌─────────────────────────────────────┐
│ python web framework              🔍│
└─────────────────────────────────────┘

About 234 results (12 ms)

Python Web Framework
https://example.com/python

Python is a programming language used
for building web applications...

--------------------------------------

FastAPI
https://fastapi.tiangolo.com

FastAPI is a modern Python framework...
```

---

# 29. FRONTEND COMPONENT RESPONSIBILITIES

### SearchBar

Responsible for:

* input
* submit
* loading state

### SearchResults

Responsible for:

* result list
* total results
* timing

### SearchResult

Responsible for:

* title
* URL
* snippet
* score if debugging mode is enabled

### Pagination

Responsible for:

* next page
* previous page
* page numbers

### LoadingState

Display while API request is running.

### ErrorState

Display when API fails.

---

# 30. FRONTEND API CLIENT

Create:

```text
frontend/lib/api.ts
```

It should contain functions such as:

```text
search(query, page, limit)
```

Do not call the backend directly from every React component.

Centralize API communication.

---

# 31. CONFIGURATION

All configurable values must live in configuration.

Examples:

```text
DATABASE_URL
API_HOST
API_PORT
CRAWLER_USER_AGENT
CRAWLER_TIMEOUT
CRAWLER_DELAY
MAX_CRAWL_PAGES
MAX_CRAWL_DEPTH
BM25_K1
BM25_B
INDEX_PATH
```

Use environment variables where appropriate.

Provide:

```text
.env.example
```

Never commit secrets.

---

# 32. LOGGING

Implement structured logging.

Important events:

```text
crawler started
URL fetched
URL rejected
robots.txt blocked
page parsed
document indexed
crawl completed
search executed
search failed
```

Avoid excessive debug logging in production.

---

# 33. ERROR HANDLING

The backend must gracefully handle:

* invalid URLs
* malformed HTML
* timeout
* DNS failure
* connection errors
* HTTP 404
* HTTP 403
* HTTP 429
* HTTP 500
* robots.txt failures
* duplicate documents
* empty queries
* malformed API requests

Never allow a single bad page to terminate the entire crawl.

---

# 34. TESTING

Tests are mandatory.

Write unit tests for:

### URL normalization

```text
test_normalize_url()
test_remove_fragment()
test_normalize_hostname()
```

### Tokenizer

```text
test_lowercase()
test_punctuation()
test_unicode()
```

### Inverted index

```text
test_add_document()
test_term_lookup()
test_term_frequency()
```

### TF-IDF

```text
test_tf()
test_idf()
test_tfidf()
```

### BM25

```text
test_bm25_score()
test_document_length_normalization()
```

### Search

```text
test_basic_search()
test_multi_term_search()
test_ranking()
```

### Crawler

```text
test_duplicate_urls()
test_max_depth()
test_max_pages()
test_robots_behavior()
```

### API

```text
test_search_endpoint()
test_empty_query()
test_invalid_request()
```

---

# 35. EVALUATION DATASET

Create a small deterministic dataset.

For example:

```text
data/
├── documents/
│   ├── python.txt
│   ├── javascript.txt
│   ├── databases.txt
│   ├── networking.txt
│   └── algorithms.txt
│
└── test_queries.json
```

`test_queries.json` should contain:

```json
[
    {
        "query": "python programming",
        "expected_documents": [
            "python.txt"
        ]
    }
]
```

Use this dataset to verify ranking behavior.

---

# 36. PERFORMANCE REQUIREMENTS — V1

For a development dataset of approximately:

```text
1,000–10,000 documents
```

target:

```text
index construction:
reasonable for local development

simple search:
< 100 ms target

BM25 search:
< 200 ms target

API overhead:
minimal
```

These are development targets, not promises.

Do not prematurely optimize.

Measure actual performance.

Create timing utilities so search latency can be measured.

---

# 37. INDEX PERSISTENCE

The index should survive application restarts.

Initially implement a simple index persistence mechanism.

For example:

```text
data/
└── index/
    ├── inverted_index
    ├── document_stats
    └── metadata
```

Use a serialization format appropriate for the implementation.

Create an abstraction:

```text
IndexStorage
├── save()
├── load()
├── add()
├── remove()
└── clear()
```

This allows a future storage engine to replace the initial implementation.

---

# 38. INCREMENTAL INDEXING

The index should eventually support:

```text
add document
update document
delete document
```

Do not require rebuilding the entire index every time one page changes.

For V1, a simple rebuild strategy is acceptable if incremental updates make the implementation unnecessarily complex.

However, design interfaces so incremental indexing can be introduced later.

---

# 39. DUPLICATE DETECTION

Use:

```text
canonical URL
```

and:

```text
content hash
```

to detect duplicates.

Example:

```text
URL A
URL B
```

may point to identical content.

Generate a hash of normalized document content.

If the hash already exists, do not index the duplicate as a completely new document.

---

# 40. SEARCH RESULT QUALITY

Ranking should eventually consider:

```text
1. Query relevance
2. Term frequency
3. Term rarity
4. Document length
5. Title match
6. Heading match
7. URL match
8. Anchor text
9. Freshness
10. Link authority
```

For V1:

```text
BM25
+
title boost
+
heading boost
+
URL boost
```

is enough.

Do not implement PageRank until the basic ranking system is working.

---

# 41. FUTURE PHASE — LINK GRAPH

Later introduce:

```text
Page A ─────► Page B
   │
   └────────► Page C

Page B ─────► Page C
```

Store the graph.

Then calculate an authority score such as PageRank.

Future ranking:

```text
final_score =
    relevance_score * relevance_weight
    +
    authority_score * authority_weight
```

Keep this separate from BM25.

---

# 42. FUTURE PHASE — SEMANTIC SEARCH

After keyword search works correctly, add semantic search.

Pipeline:

```text
Document
   ↓
Embedding model
   ↓
Vector
   ↓
Vector database
```

Query:

```text
User query
   ↓
Embedding
   ↓
Vector similarity
   ↓
Semantic candidates
```

Do not add embeddings to V1.

---

# 43. FUTURE PHASE — HYBRID SEARCH

Eventually support:

```text
                    QUERY
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
          BM25             Vector Search
             │                 │
             ▼                 ▼
       Keyword Results   Semantic Results
             │                 │
             └────────┬────────┘
                      ▼
                 Merge Results
                      │
                      ▼
                 Re-rank
                      │
                      ▼
                  Top Results
```

Possible hybrid score:

```text
hybrid_score =
    alpha * keyword_score
    +
    beta * semantic_score
```

Make `alpha` and `beta` configurable.

---

# 44. FUTURE PHASE — LLM FEATURES

Do NOT use an LLM as the fundamental search engine.

LLMs can later be added for:

* query rewriting
* query expansion
* result summarization
* answer generation
* intent detection
* spelling correction
* semantic query understanding

The underlying retrieval system must remain independent.

Architecture later:

```text
User
 ↓
Query Understanding
 ↓
Retriever
 ├── BM25
 └── Vector Search
 ↓
Re-ranker
 ↓
Results
 ↓
Optional LLM Answer
```

---

# 45. FUTURE SCALABILITY

Design the project so the following can eventually replace V1 components:

```text
SQLite
    ↓
PostgreSQL

In-memory index
    ↓
OpenSearch / Elasticsearch / custom distributed index

Local queue
    ↓
Redis / Kafka

Single crawler
    ↓
Multiple crawler workers

Single API
    ↓
Multiple API instances

Local vector index
    ↓
Distributed vector database
```

Do not implement these now.

Create interfaces where appropriate.

---

# 46. DOCKER

Create a basic Docker setup.

Initially:

```text
backend
frontend
```

Do not introduce unnecessary services.

Later Docker Compose can include:

```text
backend
frontend
postgres
redis
vector-db
```

only when needed.

---

# 47. README

The README must explain:

```text
Project overview
Architecture
Tech stack
Folder structure
How to install
How to run backend
How to run frontend
How to add documents
How to start crawling
How indexing works
How search works
How ranking works
How to run tests
Performance testing
Future roadmap
```

Also include a section:

```text
How a search query works
```

with the complete pipeline.

---

# 48. DEVELOPMENT PHASES

Do NOT attempt to implement everything at once.

Implement in these phases.

## Phase 1

Create project structure.

Set up:

```text
Python
FastAPI
Next.js
SQLite
pytest
```

Verify both frontend and backend run.

---

## Phase 2

Implement document storage.

Build:

```text
Document model
Database
Repositories
Document API
```

Verify documents can be inserted and retrieved.

---

## Phase 3

Implement tokenizer.

Write tests.

Do not continue until tokenizer tests pass.

---

## Phase 4

Implement inverted index.

Use manually created documents.

Verify:

```text
term → documents
```

works correctly.

---

## Phase 5

Implement simple search.

Verify:

```text
query
 ↓
tokens
 ↓
posting lists
 ↓
candidate documents
 ↓
results
```

---

## Phase 6

Implement TF-IDF.

Compare rankings against expected behavior.

---

## Phase 7

Implement BM25.

Write unit tests with known inputs.

---

## Phase 8

Implement title/body/heading weighting.

---

## Phase 9

Implement snippet generation.

---

## Phase 10

Implement crawler.

Start with:

```text
max_pages = 10
max_depth = 1
```

Test carefully.

---

## Phase 11

Connect crawler → parser → document store → index.

---

## Phase 12

Build frontend.

Connect:

```text
Next.js → FastAPI → Search Engine
```

---

## Phase 13

Increase crawl limits.

Test with:

```text
100 pages
500 pages
1,000 pages
```

Measure:

```text
crawl speed
index size
search latency
memory usage
```

---

# 49. IMPLEMENTATION RULE

At every phase:

1. Explain what is being built.
2. Explain why it exists.
3. Implement it.
4. Write tests.
5. Run tests.
6. Fix failures.
7. Only then move to the next phase.

Do NOT generate the entire project in one enormous code dump.

Build incrementally.

---

# 50. CODE QUALITY

Use:

* type hints
* clear function names
* small functions
* classes only when they provide real value
* docstrings for important components
* meaningful error messages
* dependency injection where useful
* separation of concerns

Avoid:

* giant classes
* global mutable state
* duplicated logic
* magic numbers
* unnecessary abstractions
* unnecessary frameworks

---

# 51. LEARNING REQUIREMENT

Whenever implementing an important algorithm, explain it before implementing it.

For example, before implementing an inverted index explain:

```text
What problem does it solve?
Why is scanning every document slow?
What is a posting list?
What is term frequency?
What is document frequency?
```

Before BM25:

```text
Why isn't raw keyword frequency enough?
What problem does TF-IDF solve?
Why does BM25 normalize document length?
```

The goal is for the developer to understand the implementation rather than merely copy it.

---

# 52. DEBUG MODE

Implement a development/debug mode.

For a query such as:

```text
python web framework
```

debug output should optionally show:

```text
Query tokens:
python
web
framework

Candidate documents:
12
24
51

BM25:
document 12 = 8.21
document 24 = 5.72
document 51 = 3.19

Title boost:
document 12 = +2.0

Final:
document 12 = 10.21
```

This is extremely useful for understanding and tuning ranking.

Do not expose debug information in normal production responses.

---

# 53. OBSERVABILITY

Track:

```text
crawl pages/sec
documents indexed
index size
search latency
candidate count
ranking latency
API latency
crawler errors
HTTP status distribution
```

This information should be available through logs or a development stats endpoint.

---

# 54. SECURITY

The API must validate:

* URLs
* query lengths
* request sizes
* crawl limits

Prevent the crawler from being used to request arbitrary internal resources.

Implement protections against:

* localhost crawling
* private IP ranges
* internal network addresses
* excessively large responses
* infinite redirects

This becomes especially important if the crawler is ever exposed publicly.

---

# 55. DEFINITION OF DONE — V1

V1 is complete when all of the following work:

```text
[ ] Python backend runs
[ ] Next.js frontend runs
[ ] SQLite database works
[ ] Documents can be manually added
[ ] Documents can be retrieved
[ ] Tokenizer works
[ ] Inverted index works
[ ] Search works
[ ] TF-IDF works
[ ] BM25 works
[ ] Title weighting works
[ ] Snippets work
[ ] Crawler works
[ ] robots.txt is respected
[ ] URL normalization works
[ ] Duplicate detection works
[ ] Search API works
[ ] Crawl API works
[ ] Frontend search works
[ ] Pagination works
[ ] Error handling works
[ ] Unit tests pass
[ ] README is complete
```

---

# 56. V1 SEARCH FLOW

The final V1 query flow should be:

```text
USER
 │
 │ "python web framework"
 ▼
NEXT.JS
 │
 │ GET /search?q=...
 ▼
FASTAPI
 │
 ▼
QUERY PARSER
 │
 ├── normalize
 ├── tokenize
 └── parse operators
 │
 ▼
CANDIDATE RETRIEVAL
 │
 └── inverted index
 │
 ▼
BM25 RANKING
 │
 ├── term relevance
 ├── document length
 ├── title boost
 ├── heading boost
 └── URL boost
 │
 ▼
SNIPPET GENERATOR
 │
 ▼
SEARCH API RESPONSE
 │
 ▼
NEXT.JS
 │
 ▼
USER
```

---

# 57. DO NOT DO THESE THINGS

Do NOT:

* build the entire system in one file
* use Elasticsearch as the initial implementation
* use an LLM to rank all results
* use an LLM to replace BM25
* crawl the entire internet
* ignore robots.txt
* create an uncontrolled crawler
* hard-code configuration
* put database code inside React
* put crawler code inside FastAPI routes
* skip tests
* skip documentation
* optimize prematurely
* add Redis/Postgres/vector DB before they are needed
* generate huge amounts of code without testing

---

# 58. FUTURE ROADMAP

After V1:

```text
V1
│
├── crawler
├── parser
├── inverted index
├── BM25
├── API
└── frontend
        │
        ▼
V2
├── PageRank
├── better crawling
├── incremental indexing
├── improved snippets
└── search operators
        │
        ▼
V3
├── semantic embeddings
├── vector search
└── hybrid search
        │
        ▼
V4
├── distributed crawling
├── distributed indexing
├── caching
└── scalable infrastructure
        │
        ▼
V5
├── query understanding
├── re-ranking
├── LLM-assisted search
└── answer generation
```

---

# 59. FIRST TASK

Start ONLY with Phase 1.

Do not implement the crawler, BM25, semantic search, or LLM features yet.

Your first task is:

1. Create the complete project structure.
2. Set up the Python backend.
3. Set up FastAPI.
4. Set up the Next.js TypeScript frontend.
5. Set up SQLite.
6. Set up pytest.
7. Create configuration management.
8. Create the initial README.
9. Create a basic `/health` endpoint.
10. Create a basic frontend page that confirms the frontend is running.
11. Make sure the frontend can successfully communicate with the backend.
12. Run the backend tests.
13. Run the frontend build.
14. Report exactly what was created and whether each verification step passed.

Do NOT move to Phase 2 until Phase 1 is working.

When moving to subsequent phases, implement one phase at a time and verify it before continuing.
