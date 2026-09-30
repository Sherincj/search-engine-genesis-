# Semantic Search (Phase 14)

Phase 14 introduces a vector-based semantic search engine alongside the existing BM25 lexical engine. 
The two engines operate independently to allow measurable comparison before any hybrid fusion is attempted.

## Architecture

The semantic search pipeline operates as follows:

```text
                  DOCUMENT INGESTION
                           │
                           ▼
                  build_embedding_text()  (Title + Body)
                           │
                           ▼
                  SentenceTransformerProvider
                           │
                           ▼
                      VectorIndex (In-Memory)
                           │
                           ▼
                  VectorStorage (Disk Persistence)

                         QUERY
                           │
                           ▼
                  SentenceTransformerProvider
                           │
                           ▼
                      Query Vector
                           │
                           ▼
                  VectorIndex.search()
                           │
                           ▼
                   Cosine Similarity
                           │
                           ▼
                   Top-K Documents
```

## Embedding Model

By default, the engine uses the `all-MiniLM-L6-v2` model from the `sentence-transformers` library.
This is a small, efficient model that produces vectors of dimension 384.

Configuration is available in `app.config.settings`:
- `EMBEDDING_ENABLED`: Set to `True` to enable semantic search endpoints.
- `EMBEDDING_MODEL`: The model name (default: `all-MiniLM-L6-v2`).
- `EMBEDDING_BATCH_SIZE`: Batch size for generating embeddings (default: 32).
- `EMBEDDING_NORMALIZE`: Whether to L2-normalize vectors (default: `True`).
- `VECTOR_INDEX_PATH`: Persistent NPZ/JSON vector-index location.
- `VECTOR_INDEX_VERSION`: Storage format version; incompatible indexes are rejected.

## Vector Index

The custom `VectorIndex` implements exact (brute-force) nearest neighbor search using cosine similarity.
It has an O(N × D) retrieval complexity, where N is the number of documents and D is the embedding dimension.

Cosine similarity is calculated directly as `A · B / (|A| × |B|)`. The API
returns that unmodified cosine-similarity value as `score`, in `[-1, 1]`.

Approximate nearest neighbor (ANN) algorithms (like HNSW or FAISS) are intentionally avoided in this phase
to establish a correct and deterministic baseline.

## Lexical vs. Semantic

Why do we need semantic search?

Lexical search (BM25) looks for exact or partial word matches. A query for `car` will not match a document
that only contains the word `automobile`, even though they share the same meaning.

Semantic search converts text into dense mathematical vectors where concepts with similar meanings are 
placed close together in vector space. The `SemanticSearchEngine` uses the cosine similarity between the query 
vector and document vectors to find conceptually relevant documents, even if exact keywords do not match.

## APIs

The semantic search engine is exposed at a separate endpoint to ensure the existing lexical search remains untouched:

```text
GET /search/semantic?q=python&limit=10&page=1
```

Returns:
```json
{
  "query": "python",
  "results": [
    {
      "document_id": 1,
      "score": 0.8523,
      "title": null,
      "url": null
    }
  ],
  "total": 1,
  "took_ms": 15.2,
  "embedding_ms": 12.0,
  "search_ms": 1.1
}
```
*Note: The `score` returned by semantic search is the actual cosine similarity in the range `[-1, 1]`.*

## CLI Tools

### Building the Vector Index
To generate embeddings for all documents stored in the database:

```bash
python -m scripts.build_vector_index
```
This script will load all documents, encode them in batches using the configured model, and save the resulting index to `data/vector_index/`.

### Evaluation
The Phase 13 evaluation framework now supports semantic search via the `--engine` flag.

Evaluate semantic search:
```bash
python -m scripts.evaluate_search --engine semantic
```

Compare semantic search against the lexical baseline:
```bash
python -m scripts.evaluate_search --engine compare
```

The comparison uses the same Phase 13 judgments and prints aggregate and
category-level results for the two independent engines. It does not combine
their scores.
