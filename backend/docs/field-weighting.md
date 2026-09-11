# Field Weighting

Phase 8 introduces field-aware indexing and field-weighted BM25 scoring.

## Concept

Instead of treating a document as a single undifferentiated block of text, the search engine now indexes the `title` and `body` independently.

```text
Document
 ├── Title → Tokenizer → Title Index (InvertedIndex)
 └── Body  → Tokenizer → Body Index  (InvertedIndex)
```

During ranking, the relevance of a document is the sum of its field-specific BM25 scores, each scaled by a configurable weight.

## Architecture

To preserve the battle-tested `InvertedIndex` from Phase 4 without breaking its invariants, Phase 8 introduces a `FieldIndex` wrapper.

### `FieldIndex`
The `FieldIndex` contains exactly two `InvertedIndex` instances:
- `title_index`
- `body_index`

It provides unified methods to:
- Add a document (`add_document(id, title_tokens, body_tokens)`)
- Retrieve candidates across all fields (OR semantics)
- Query field-specific statistics (DF, length, average length)

### Storage & Versioning
The index storage format (JSON) is updated to **Version 2**.
An old Phase 7 index (Version 1, lacking the `"index_version"` key) will trigger an `IndexStorageError` on load, forcing a rebuild to ensure titles are correctly tokenised and indexed.

## Field-Weighted BM25 Ranking

The ranking formula applies standard BM25 independently to each field:

```
Score = (W_title × BM25_title) + (W_body × BM25_body)
```

### Parameters
The weights are configured in `app.config.settings`:
- `BM25_TITLE_WEIGHT` (default: `2.0`)
- `BM25_BODY_WEIGHT` (default: `1.0`)
- `BM25_K1` (default: `1.2`)
- `BM25_B` (default: `0.75`)

### Characteristics
1. **Independent Statistics**: `avgdl` (average document length) and document length (`|D|`) are computed per-field. A 10-word title is long for a title, but short for a body.
2. **Missing Fields**: If a document has no title, its title length is 0, its title score is 0, and the ranking degrades gracefully to the body score.
3. **Weight Pruning**: If a weight is configured to `0`, the ranker completely skips computing BM25 for that field, saving CPU.

## Candidate Retrieval

Candidate retrieval (`retrieve_candidates`) operates on the `FieldIndex`.
A document is retrieved if any query term appears in **any** field (OR semantics). The union of document IDs is then passed to the ranker for precise scoring.
