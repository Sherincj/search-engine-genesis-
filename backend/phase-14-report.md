# Phase 14 — Vector Search Foundation

Status: COMPLETE

## Tests

- Previous test functions: 649
- New Phase 14 test functions: 87
- Total collected tests: 736
- Result: 731 passed, 5 skipped (optional real-model tests) in 12.87 seconds

The skipped tests require `RUN_EMBEDDING_INTEGRATION_TESTS=1`; normal tests
remain fast and offline-friendly through deterministic fake providers.

## Embedding model

- Model: `all-MiniLM-L6-v2`
- Dimension: 384
- Normalization: enabled by default
- Batch size: configurable with `EMBEDDING_BATCH_SIZE`

## Architecture implemented

`Document -> build_embedding_text(title + body) -> EmbeddingProvider ->
VectorIndex -> VectorStorage`

Semantic query retrieval is independent of the lexical pipeline. It performs
exact O(N × D) cosine-similarity search and returns score-descending,
document-ID-ascending ties. No hybrid score fusion, reranking, ANN, or vector
database is used.

The persistent index stores model name, dimension, normalization, document
count, and index version. Mismatches are rejected. Document creation and
deletion update the persistent vector index when `EMBEDDING_ENABLED=true`.

## Search API

`GET /search/semantic?q=...&limit=...&page=...` returns cosine scores,
document title, and URL. `/search` remains the unmodified BM25 endpoint.

## Evaluation results

The shared Phase 13 dataset was evaluated with the configured real model.

| Metric | BM25 | Semantic |
| --- | ---: | ---: |
| Precision@1 | 0.8500 | 0.8500 |
| Precision@3 | 0.6167 | 0.7667 |
| Precision@5 | 0.5000 | 0.5200 |
| Recall@5 | 0.7444 | 0.7903 |
| F1@5 | 0.5730 | 0.6028 |
| MRR | 0.8750 | 0.8750 |
| MAP | 0.7358 | 0.8060 |
| NDCG@5 | 0.7530 | 0.8327 |
| NDCG@10 | 0.7794 | 0.8483 |

Semantic retrieval improved the aggregate ranking metrics on this fixed
dataset, particularly Precision@3, MAP, and NDCG. BM25 remained stronger for
the exact-phrase category (NDCG@5: 0.9877 versus 0.9449). Both engines score
the existing zero-result category at zero; results have intentionally not been
combined.

## Known limitations

- Exact search is linear in document count and is not appropriate for a large
  corpus.
- Building or querying a new model requires its local model files to be
  available.
- Semantic results do not generate semantic-specific snippets.
- Hybrid retrieval is deliberately deferred to Phase 15.
