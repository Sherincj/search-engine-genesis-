# Phrase Search (Phase 11)

Phase 11 introduces the ability to search for exact phrases using double quotes, e.g., `"python development"`.

## Architecture

Phrase matching uses the existing `positions` list stored for each term in the inverted index `Posting`. The system does not require any new index format or storage—it just computes adjacency during query time.

```
User Query: python "web development"
    ↓
query_parser.py 
    Parses into ParsedQuery(terms=["python"], phrases=[("web", "development")])
    ↓
candidate_retrieval.py 
    Fetches documents matching ANY term or phrase term (OR semantics)
    ↓
phrase_matching.py 
    Filters and extracts positions where terms appear consecutively in the same field
    ↓
bm25_ranker.py 
    Scores documents using standard BM25
    ↓
search_engine.py 
    Adds `PHRASE_BOOST` to documents that matched phrases
    Re-sorts the results
    ↓
snippet_generator.py 
    Selects snippet windows that contain the exact phrase
```

## Features

### 1. Two-Pointer Adjacency Matching
For a phrase to match, its terms must appear sequentially in the original text, with no other words in between. For example, `"python development"` matches:
- `Advanced python development guide`
But does NOT match:
- `Python for web development`
- `Development in Python`

### 2. Field Awareness
Phrase positions are checked per field. If `python` is the last word of the title, and `development` is the first word of the body, it is **not** a phrase match, even if they appear sequentially when the text is concatenated.

### 3. Rarest-Term Optimization
To speed up phrase checking, the engine finds the least frequent term in the phrase and only checks documents containing that rare term, drastically reducing the search space.

### 4. Phrase Boosting vs Filtering
- **Pure Phrase Queries**: e.g., `"python development"`. Documents that don't contain the exact phrase are immediately filtered out.
- **Mixed Queries**: e.g., `learn "python development"`. Documents containing "learn" but not the phrase will still be returned, but documents containing the phrase receive a massive `PHRASE_BOOST` (default 2.0) added to their BM25 score, ensuring they rank first.

### 5. Snippet Preference
The snippet generator receives the structured query and actively scans windows for the phrase sequence using regex, ensuring the user sees the highlighted phrase in context.

## Configuration

In `app/config/settings.py`:
- `PHRASE_BOOST` (default 2.0) — The additive score boost applied to documents for each matched phrase.
