# Query Processing (Phase 12)

Phase 12 introduces advanced lexical query processing to understand the structure and weighting of user queries. It improves search quality without introducing semantic or vector search.

## Architecture

```text
User Query
    ↓
QueryParser (Extracts phrases and loose terms)
    ↓
QueryProcessor
    ├── Deduplicates phrases
    ├── Counts global term frequencies
    └── Calculates sublinear term weights
    ↓
ProcessedQuery
    ↓
SearchEngine (BM25 + Phrase Boost)
    ↓
SnippetGenerator
```

## Features

### 1. Global Term Frequency & Weighting
Instead of blindly duplicating candidate retrieval for repeated terms (e.g., `python python development`), the processor counts frequencies globally across all tokens. 
Term weights are assigned sublinearly using a logarithmic formula:
`weight = min(1 + ln(frequency), 3.0)`

### 2. Backward Compatible Unique Term Scoring
For frequency=1, the weight is exactly `1.0`. The `SearchEngine` multiplies the standard field-weighted BM25 term score by this query weight. This ensures that unique-term queries behave exactly as they did in Phase 11.

### 3. Duplicate Phrase Handling
Phrases like `"python development" "python development"` are automatically deduplicated. The phrase boost is only applied once, preventing artificial score explosion. However, the internal terms (`python`, `development`) are still counted towards the global term frequencies.

### 4. Phrase-Only Query Filtering
A query like `"python development"` contains no "loose" terms. In these cases, the `ProcessedQuery.is_phrase_only` property informs the `SearchEngine` to strictly filter candidates that do not contain the matched phrase, rather than falling back to OR-based loose term matches.

### 5. Snippet Generator Reuse
The `SnippetGenerator` now explicitly accepts `ProcessedQuery`, ensuring the tokenization and phrase representations exactly match the scoring pipeline.

## Limitations
- **No Semantic Understanding**: The system remains purely lexical. "Python" and "Py" are completely separate.
- **Stopwords**: Stopword handling relies entirely on the underlying tokenizer config; the query processor does not maintain a separate stopword list.
