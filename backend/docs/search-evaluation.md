# Search Quality Evaluation (Phase 13)

Phase 13 introduces a deterministic search-quality evaluation framework that measures how well the search engine ranks documents for a given set of queries.

## Why Evaluation Exists

The search engine implements BM25, field weighting, phrase search, and query weighting. But without formal measurement, there is no way to answer:

- Did a change improve search results?
- Are phrase queries performing better than keyword queries?
- Did query weighting help or hurt?

This framework provides that measurement system.

## Architecture

```text
Evaluation Dataset (documents.json + search_quality.json)
    ↓
Dataset Loader (validates structure)
    ↓
Isolated FieldIndex (temporary, not production)
    ↓
SearchEngine (same code as production)
    ↓
Evaluator (runs queries, collects results)
    ↓
Metrics (pure mathematical functions)
    ↓
Reporter (terminal / JSON / comparison)
    ↓
Baseline (saved metrics for regression tracking)
```

## Dataset Format

### `evaluation_data/documents.json`

Array of documents:

```json
[
  {
    "id": 1,
    "url": "https://eval.test/doc1",
    "title": "Python Programming Guide",
    "text": "Python is a versatile programming language..."
  }
]
```

### `evaluation_data/search_quality.json`

Queries with relevance judgments:

```json
{
  "queries": [
    {
      "id": "q001",
      "query": "python",
      "category": "single_term",
      "judgments": {"1": 3, "5": 3, "8": 2}
    }
  ]
}
```

## Relevance Grades

| Grade | Meaning |
|-------|---------|
| 0 | Irrelevant |
| 1 | Weakly relevant |
| 2 | Relevant |
| 3 | Highly relevant |

Judgments are manually defined. They are NOT generated from the search engine itself.

## Query Categories

| Category | Description |
|----------|-------------|
| `single_term` | One keyword: `python` |
| `multi_term` | Multiple keywords: `python web development` |
| `phrase` | Quoted phrase: `"machine learning"` |
| `mixed` | Term + phrase: `python "web development"` |
| `repeated_term` | Repeated keyword: `python python` |
| `zero_result` | No matching documents expected |

## Metrics

### Precision@K

Fraction of top-K results that are relevant (binary: relevance > 0).

```
Precision@K = |relevant ∩ top-K| / K
```

### Recall@K

Fraction of all relevant documents found in top-K.

```
Recall@K = |relevant ∩ top-K| / |relevant|
```

Returns 0.0 if there are no relevant documents.

### F1@K

Harmonic mean of Precision@K and Recall@K.

```
F1@K = 2 × P × R / (P + R)
```

Returns 0.0 if both are zero.

### MRR (Mean Reciprocal Rank)

For each query, the reciprocal rank is `1/position` of the first relevant result. MRR is the mean across all queries.

### AP (Average Precision)

For each relevant result at rank k, accumulate Precision@k, then divide by the total number of relevant documents.

Returns 0.0 if there are no relevant documents.

### MAP (Mean Average Precision)

Mean of AP across all queries. Zero-result queries contribute AP=0.

### NDCG@K (Normalized Discounted Cumulative Gain)

Uses **graded** relevance:

```
DCG@K = Σ (2^rel_i - 1) / log₂(i + 1)  for i = 1..K
NDCG@K = DCG@K / IDCG@K
```

IDCG is the ideal ranking. Returns 0.0 if IDCG is zero.

## Running Evaluation

```bash
# Human-readable report
python -m scripts.evaluate_search

# Verbose (per-query details)
python -m scripts.evaluate_search --verbose

# JSON output
python -m scripts.evaluate_search --json

# Save baseline
python -m scripts.evaluate_search --save-baseline

# Compare with baseline
python -m scripts.evaluate_search --compare evaluation/baseline.json
```

## Baseline

The baseline captures metric values at a point in time:

```json
{
  "version": 1,
  "phase": 13,
  "configuration": {
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "title_weight": 2.0,
    "body_weight": 1.0,
    "phrase_boost": 2.0
  },
  "metrics": {
    "precision@1": 0.85,
    "mrr": 0.875,
    "ndcg@5": 0.753
  }
}
```

## Determinism

The evaluation is fully deterministic:
- No random sampling
- No network requests
- No production database dependency
- Same corpus always produces the same metrics

## Limitations

- The corpus is small (30 documents, 20 queries) — metrics indicate relative quality, not absolute.
- Relevance judgments are subjective and manually defined.
- The evaluation measures lexical search only — it does not test semantic understanding.
- No automated relevance judgment generation — all judgments are hand-crafted.
