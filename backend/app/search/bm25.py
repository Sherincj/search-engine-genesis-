"""
BM25 mathematical building blocks.

Pure mathematical functions — no FastAPI, SQLite, or HTTP dependencies.

Formulas
--------
IDF(t)  = log(1 + (N - DF + 0.5) / (DF + 0.5))   — BM25-smoothed IDF
BM25    = Σ IDF(t) × f(t,D)×(k1+1) / (f(t,D) + k1×(1 - b + b×|D|/avgdl))

Parameters
----------
k1 : float   — term-frequency saturation (default 1.2)
b  : float   — document-length normalization (default 0.75)
"""

from __future__ import annotations

import math


def bm25_idf(
    total_documents: int,
    document_frequency: int,
) -> float:
    """
    BM25-smoothed inverse document frequency.

    ``IDF = log(1 + (N - DF + 0.5) / (DF + 0.5))``

    Unlike the Phase 6 TF-IDF IDF (``log(N/DF)``), BM25 IDF:
    - Never becomes exactly zero (a term in every doc still has a
      small positive IDF)
    - Uses additive smoothing (+0.5) to avoid division by zero

    Parameters
    ----------
    total_documents : int
        Total indexed documents (N).  Must be ≥ 1.
    document_frequency : int
        Documents containing the term (DF).  Must be ≥ 1 and ≤ N.

    Returns
    -------
    float
        BM25-smoothed IDF (always ≥ 0).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if total_documents < 1:
        raise ValueError(
            f"total_documents must be >= 1, got {total_documents}"
        )
    if document_frequency < 1:
        raise ValueError(
            f"document_frequency must be >= 1, got {document_frequency}"
        )
    if document_frequency > total_documents:
        raise ValueError(
            f"document_frequency ({document_frequency}) cannot exceed "
            f"total_documents ({total_documents})"
        )
    return math.log(
        1.0 + (total_documents - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )


def bm25_term_score(
    term_frequency: int,
    document_length: int,
    average_document_length: float,
    idf: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """
    BM25 per-term score component.

    ``IDF × f×(k1+1) / (f + k1×(1 - b + b×|D|/avgdl))``

    Parameters
    ----------
    term_frequency : int
        Times this term appears in the document.  Must be ≥ 0.
    document_length : int
        Total tokens in the document.  Must be ≥ 0.
    average_document_length : float
        Mean document length across all indexed documents.  Must be > 0.
    idf : float
        Pre-computed BM25 IDF for this term.
    k1 : float
        Term-frequency saturation parameter.  Must be ≥ 0.
    b : float
        Document-length normalization parameter.  Must be in [0, 1].

    Returns
    -------
    float
        This term's contribution to the document's BM25 score.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    _validate_params(term_frequency, document_length,
                     average_document_length, k1, b)

    if term_frequency == 0:
        return 0.0

    tf = float(term_frequency)
    dl = float(document_length)
    avgdl = float(average_document_length)

    numerator = tf * (k1 + 1.0)
    denominator = tf + k1 * (1.0 - b + b * dl / avgdl)

    return idf * numerator / denominator


def bm25_score(
    term_frequencies: dict[str, int],
    document_length: int,
    average_document_length: float,
    idfs: dict[str, float],
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """
    Complete BM25 score for a document against a query.

    Sums per-term scores for all query terms present in the document.

    Parameters
    ----------
    term_frequencies : dict[str, int]
        ``{term: frequency}`` for query terms found in this document.
    document_length : int
        Total tokens in the document.
    average_document_length : float
        Mean document length across all indexed documents.
    idfs : dict[str, float]
        ``{term: idf}`` for each query term.
    k1 : float
        Term-frequency saturation parameter.
    b : float
        Document-length normalization parameter.

    Returns
    -------
    float
        Total BM25 relevance score for this document.
    """
    total = 0.0
    for term, freq in term_frequencies.items():
        idf = idfs.get(term, 0.0)
        total += bm25_term_score(
            term_frequency=freq,
            document_length=document_length,
            average_document_length=average_document_length,
            idf=idf,
            k1=k1,
            b=b,
        )
    return total


def _validate_params(
    term_frequency: int,
    document_length: int,
    average_document_length: float,
    k1: float,
    b: float,
) -> None:
    """Validate BM25 parameters, raising ValueError if invalid."""
    if term_frequency < 0:
        raise ValueError(
            f"term_frequency must be >= 0, got {term_frequency}"
        )
    if document_length < 0:
        raise ValueError(
            f"document_length must be >= 0, got {document_length}"
        )
    if average_document_length <= 0:
        raise ValueError(
            f"average_document_length must be > 0, got "
            f"{average_document_length}"
        )
    if k1 < 0:
        raise ValueError(f"k1 must be >= 0, got {k1}")
    if b < 0 or b > 1:
        raise ValueError(f"b must be in [0, 1], got {b}")
