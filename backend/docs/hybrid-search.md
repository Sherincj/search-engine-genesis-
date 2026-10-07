# Hybrid Search (Phase 15)

Phase 15 combines BM25 lexical search and semantic vector search into a unified hybrid ranking.

## Why Hybrid Search

BM25 excels at exact keyword matching and phrase queries. Semantic search excels at finding conceptually related documents even when exact terms don't match. Hybrid search combines both signals to produce more robust rankings.

## Architecture

```text
                         QUERY
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        BM25 Engine              Semantic Engine
              │                         │
         Top-K (50)                Top-K (50)
              │                         │
              └────────────┬────────────┘
                           │
                    Candidate Union
                           │
                    Score Normalization
                           │
                    Weighted Fusion
                           │
                    Final Ranking
```

Both the BM25 and Semantic engines remain completely independent and unchanged.

## Score Normalization

BM25 scores and cosine similarity are on different scales:
- BM25: unbounded positive values (e.g., 3.2 to 12.8)
- Cosine similarity: bounded in [-1, 1]

**Min-max normalization** transforms both to [0, 1]:

```
normalized = (score - min) / (max - min)
```

Edge cases:
- All scores equal → normalized to 1.0
- Empty input → empty output

**Important:** Normalization is query-local. A normalized score does NOT represent an absolute probability of relevance.

## Score Fusion

Uses weighted linear combination:

```
hybrid_score = lexical_weight × normalized_bm25 + semantic_weight × normalized_semantic
```

Default weights: 0.5 / 0.5 (equal influence).

## Missing Score Policy

When a document appears in only one result set:
- Document in BM25 only → semantic score = 0.0
- Document in semantic only → BM25 score = 0.0

This is applied before normalization, meaning missing candidates receive the minimum normalized score (0.0 after min-max).

## Candidate Retrieval

Both engines retrieve their top-K candidates (default: 50 each). The candidate union is then scored and ranked by the hybrid score.

Documents appearing in both result sets benefit from both signals. Documents in only one set can still rank highly if their single signal is strong enough.

## Phrase Queries

The hybrid engine consumes existing BM25 results, which include phrase boosting from Phase 11. Phrase queries like `"web development"` retain their lexical precision through the BM25 component.

## Configuration

Settings in `app/config/settings.py`:

```python
HYBRID_LEXICAL_WEIGHT = 0.5    # BM25 influence
HYBRID_SEMANTIC_WEIGHT = 0.5   # Semantic influence  
HYBRID_CANDIDATE_LIMIT = 50    # Candidates per engine
```

Weights must be non-negative. For interpretability, they should sum to 1.0.

## API

```text
GET /search/hybrid?q=python&limit=10&page=1&debug=false
```

The `debug=true` parameter includes raw lexical and semantic scores in the response.

## Evaluation CLI

```bash
# Evaluate hybrid search
python -m scripts.evaluate_search --engine hybrid

# Custom weights
python -m scripts.evaluate_search --engine hybrid --lexical-weight 0.75 --semantic-weight 0.25

# Weight sweep (evaluates 5 configurations)
python -m scripts.evaluate_search --engine hybrid --weight-sweep
```

## Weight Sweep

The sweep evaluates these configurations:
| Lexical | Semantic | Note |
|---------|----------|------|
| 1.00 | 0.00 | Pure BM25 (sanity check) |
| 0.75 | 0.25 | BM25-dominant |
| 0.50 | 0.50 | Equal (default) |
| 0.25 | 0.75 | Semantic-dominant |
| 0.00 | 1.00 | Pure semantic (sanity check) |

## Future Fusion Strategies

The `FusionStrategy` protocol allows alternative fusion methods without changing candidate retrieval. Potential future strategies include:
- **Reciprocal Rank Fusion (RRF)**: Combines rankings rather than raw scores
- **Learned Fusion**: Trains weights from click data

These are NOT implemented in Phase 15.

## Limitations

- Query-local normalization (scores are not globally comparable)
- Fixed weights (no automatic tuning)
- Exact vector search (O(N×D), no ANN)
- No learned ranking or reranking
- No cross-encoder refinement
