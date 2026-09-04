# Inverted Index — Technical Documentation

## What Is an Inverted Index?

An inverted index reverses the relationship between documents and terms.

**Without an inverted index** (forward index):

```
Document 1 → [python, web, framework]
Document 2 → [javascript, web]
Document 3 → [python, database]
```

Searching for `python` requires scanning **every document**.

**With an inverted index:**

```
python     → [Document 1, Document 3]
web        → [Document 1, Document 2]
javascript → [Document 2]
database   → [Document 3]
framework  → [Document 1]
```

Searching for `python` is a **direct lookup** — O(1).

---

## Why It Is Needed

The inverted index is the core data structure that makes search fast.
Every major search engine (Google, Elasticsearch, Lucene, etc.) is built
on some form of inverted index.

Future ranking algorithms (TF-IDF, BM25) need the information stored
in the index to calculate relevance scores.

---

## Architecture

```
Document
   ↓
DocumentIndexer
   ↓
Tokenizer  (Phase 3)
   ↓
Token Sequence
   ↓
InvertedIndex
   ↓
IndexStorage  (JSON persistence)
```

### Module structure

```
app/indexing/
├── __init__.py           # Public API exports
├── inverted_index.py     # Core InvertedIndex + Posting
├── index_builder.py      # Batch construction helper
├── index_storage.py      # JSON-based persistence
└── document_indexer.py   # Document → tokenise → index pipeline
```

---

## Posting Lists

Each term maps to a **posting list** — a sequence of `Posting` objects:

```python
@dataclass
class Posting:
    document_id: int      # Which document
    term_frequency: int   # How many times the term appears
    positions: list[int]  # Where the term appears (token offsets)
```

### Example

Given these documents:

```
Document 1: "Python web Python"
Document 2: "JavaScript web"
```

The index stores:

```
python:
    Posting(document_id=1, term_frequency=2, positions=[0, 2])

javascript:
    Posting(document_id=2, term_frequency=1, positions=[0])

web:
    Posting(document_id=1, term_frequency=1, positions=[1])
    Posting(document_id=2, term_frequency=1, positions=[1])
```

### Ordering

- Posting lists are **sorted by document ID** (ascending).
- Position lists are **sorted in ascending order** (token offsets).

---

## Term Frequency (TF)

Term frequency is the number of times a term appears **in a specific document**.

```
Document 1: "Python web Python"

TF("python", doc 1) = 2
TF("web", doc 1) = 1
```

TF is stored directly in each posting — **not recalculated during search**.

---

## Document Frequency (DF)

Document frequency is the number of **distinct documents** containing a term.

```
Document 1: "Python web Python"
Document 2: "Python JavaScript"
Document 3: "web JavaScript"

DF("python") = 2       (docs 1, 2)
DF("web") = 2           (docs 1, 3)
DF("javascript") = 2    (docs 2, 3)
```

Even though `python` appears twice in Document 1, it contributes **only one**
to DF for that document.

---

## Document Length

Document length = number of tokens produced by the tokeniser.

```
"Python is a programming language"
  → ["python", "is", "a", "programming", "language"]
  → document_length = 5
```

This uses the Phase 3 tokeniser — **not** `text.split()`.

---

## Average Document Length

```
average_document_length = total_tokens_across_all_docs / total_documents
```

If the index is empty, average document length = 0 (no division by zero).

---

## Token Positions

Positions record where each term appears as a zero-indexed token offset:

```
"Python is popular for web development"
  pos:  0      1  2       3   4   5

python → positions = [0]
web    → positions = [4]
```

Positions enable future:
- **Phrase search**: `"python web"` — terms must be adjacent
- **Proximity search**: terms within N positions of each other
- **Snippet generation**: find where terms appear for highlighting

---

## Index Statistics

The index maintains global statistics for future ranking:

| Statistic                  | Used by  | Method                           |
|----------------------------|----------|----------------------------------|
| Total documents (N)        | TF-IDF   | `get_total_documents()`          |
| Document frequency (DF)    | TF-IDF   | `get_document_frequency(term)`   |
| Document length (\|D\|)    | BM25     | `get_document_length(doc_id)`    |
| Average doc length (avgdl) | BM25     | `get_average_document_length()`  |
| Term frequency (TF)        | Both     | `posting.term_frequency`         |

---

## Index Persistence

The index is persisted as JSON for V1 — human-readable and debuggable.

### Atomic writes

To avoid corruption on interrupted saves:

```
Write to index.json.tmp → flush → atomic rename to index.json
```

A failed save never corrupts the previously valid index.

### Storage API

```python
storage = IndexStorage("data/index")
storage.save(index)          # Persist to disk
loaded = storage.load()      # Restore from disk
storage.exists()             # Check if file exists
storage.delete()             # Remove from disk
```

### Round-trip guarantee

After `save()` then `load()`, the loaded index contains equivalent:
- Postings (terms, doc IDs, TF, positions)
- Document lengths
- Document frequencies
- Total document count
- Average document length

---

## Source of Truth

```
SQLite  =  source of truth for documents
Index   =  derived search structure
```

If the index is deleted, documents remain in SQLite and the index
can be rebuilt. The index is **derived data**.

---

## How BM25 Will Use the Index (Future)

BM25 scoring for a query term *t* in document *d*:

```
score(t, d) = IDF(t) × (TF(t,d) × (k₁ + 1)) / (TF(t,d) + k₁ × (1 - b + b × |d|/avgdl))
```

Where:
- `TF(t,d)` → `posting.term_frequency`
- `IDF(t)` → derived from `DF(t)` and `N`
- `|d|` → `get_document_length(doc_id)`
- `avgdl` → `get_average_document_length()`
- `N` → `get_total_documents()`
- `DF(t)` → `get_document_frequency(term)`

All inputs are stored in the Phase 4 index. BM25 implementation is Phase 7.

---

## Validation

The index has a `validate()` method that detects inconsistencies:

```python
errors = index.validate()
if errors:
    for e in errors:
        print(e)
```

Checks:
- TF matches the number of positions
- Positions are sorted
- No empty posting lists
- All posted documents have length entries
- doc_terms ↔ postings consistency

---

## What Has NOT Been Implemented

- **Search API** — `/search` remains a 501 placeholder
- **TF-IDF scoring** — data is stored; algorithm is Phase 6
- **BM25 scoring** — data is stored; algorithm is Phase 7
- **Crawler** — no URL fetching
- **Vector search** — no embeddings, no vector store
