# TF-IDF Ranking — Technical Documentation

## Overview

Phase 6 replaces the Phase 5 baseline ranking with **TF-IDF cosine similarity**.

```
Phase 5:  matched terms + raw frequency → simple integer score
Phase 6:  TF-IDF vectors + cosine similarity → float score in [0, 1]
```

---

## Term Frequency (TF)

```
TF(t,d) = 1 + log(freq(t,d))
```

Where `freq(t,d)` is the number of times term *t* appears in document *d*.

### Why logarithmic scaling?

Raw TF treats a term appearing 100 times as 100× more important than
appearing once.  This is almost never true — the first occurrence is the
most important.

| freq | Raw TF | Log TF        |
|------|--------|---------------|
| 1    | 1      | 1.0           |
| 2    | 2      | 1.69          |
| 10   | 10     | 3.30          |
| 100  | 100    | 5.61          |
| 1000 | 1000   | 7.91          |

Logarithmic TF compresses the scale so that high-frequency terms don't
dominate.

---

## Inverse Document Frequency (IDF)

```
IDF(t) = log(N / DF(t))
```

Where:
- `N` = total indexed documents
- `DF(t)` = number of documents containing term *t*

### Why IDF matters

A term appearing in **every** document provides no discriminating power:

```
N = 100, DF = 100 → IDF = log(1) = 0
```

A term appearing in **one** document is highly informative:

```
N = 100, DF = 1 → IDF = log(100) ≈ 4.61
```

IDF is what makes rare terms like "distributed" more valuable than
common terms like "the" in search ranking.

---

## TF-IDF

```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

The combined weight balances:
- **How often** the term appears in this document (TF)
- **How rare** the term is across all documents (IDF)

Example:

```
"python" appears 3 times in doc 1
"python" appears in 50/100 documents

TF = 1 + log(3) ≈ 2.10
IDF = log(100/50) = log(2) ≈ 0.69
TF-IDF ≈ 1.45
```

vs.

```
"distributed" appears 1 time in doc 1
"distributed" appears in 2/100 documents

TF = 1 + log(1) = 1.0
IDF = log(100/2) = log(50) ≈ 3.91
TF-IDF ≈ 3.91
```

The rare term "distributed" has a higher TF-IDF weight even with
lower term frequency.

---

## Cosine Similarity

```
cos(Q, D) = (Q · D) / (|Q| × |D|)
```

Where:
- `Q` = query TF-IDF vector
- `D` = document TF-IDF vector
- `Q · D` = dot product (sum of products of shared terms)
- `|Q|` = magnitude of query vector (sqrt of sum of squared weights)
- `|D|` = magnitude of document vector

### Why cosine similarity?

Cosine similarity provides **document length normalization**.

Consider two documents:
- Doc A: "python python python python python" (5 tokens)
- Doc B: "python" (1 token)

Both are entirely about "python".  Without normalization, Doc A would
score 5× higher.  Cosine similarity measures the **direction** of the
vector, not its magnitude, so both documents score equally when their
entire content is the query term.

### Score range

```
cos = 1.0  → identical direction (perfect match)
cos = 0.0  → no shared terms (orthogonal vectors)
0 < cos < 1 → partial overlap
```

---

## Query Vector

For unique query terms, query TF is always 1:

```
query_weight(term) = TF(1) × IDF(term) = 1 × IDF(term) = IDF(term)
```

Duplicate query terms are deduplicated — "python python" is the same
as "python".

---

## Document Vector

For each document, the TF-IDF vector includes **all indexed terms**
(not just query terms).  This ensures the document magnitude is
correct for cosine similarity.

```
doc_vector = {term: TF-IDF(term, doc) for term in document_terms}
```

---

## Ranking Order

```
1. Cosine similarity — descending
2. Document ID — ascending (deterministic tie-breaker)
```

---

## Phase 5 vs Phase 6

| Aspect | Phase 5 | Phase 6 |
|--------|---------|---------|
| Score type | Integer | Float [0, 1] |
| Primary criterion | Matched term count | Cosine similarity |
| Secondary | Raw total TF | Document ID |
| Term rarity | Not considered | IDF weighting |
| Document length | Not normalized | Cosine normalization |
| Formula | count + sum(TF) | TF-IDF × cosine |

---

## Architecture

The ranking change is isolated to the scoring layer:

```
QueryParser          (unchanged — Phase 3)
      ↓
CandidateRetrieval   (unchanged — Phase 4/5)
      ↓
TF-IDF Ranker        (NEW — Phase 6)
  ├── tfidf.py       (math: TF, IDF, TF-IDF, cosine)
  └── tfidf_ranker.py (scoring pipeline)
      ↓
SearchEngine         (updated to use TF-IDF ranker)
      ↓
SearchService        (updated score field)
```

---

## API Impact

The API contract is unchanged:

```json
{
  "query": "python",
  "total": 3,
  "page": 1,
  "limit": 10,
  "results": [
    {
      "document_id": 5,
      "title": "Python Guide",
      "url": "https://example.com/python",
      "score": 0.7423
    }
  ],
  "took_ms": 1.42
}
```

The only visible change is that `score` is now a floating-point
TF-IDF cosine similarity instead of an integer matched-term count.

---

## Limitations

Phase 6 does **not** implement:

- **BM25** — no document-length tuning parameters (Phase 7)
- **Phrase search** — "python web" matches either term independently
- **Snippet generation** — no highlighted excerpts
- **Semantic search** — no embeddings or vectors
- **Fuzzy search** — exact matches only
- **PageRank** — no link analysis
- **Crawler** — no URL fetching
