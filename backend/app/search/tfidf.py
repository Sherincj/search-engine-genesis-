"""
TF-IDF mathematical building blocks.

Pure mathematical functions — no FastAPI, SQLite, or HTTP dependencies.

Formulas
--------
TF(t,d)   = 1 + log(freq(t,d))           — logarithmic term frequency
IDF(t)    = log(N / DF(t))                — inverse document frequency
TF-IDF    = TF × IDF                      — combined weight
cos(Q,D)  = (Q·D) / (|Q| × |D|)          — cosine similarity
"""

from __future__ import annotations

import math


def term_frequency(frequency: int) -> float:
    """
    Logarithmic term frequency.

    ``TF = 1 + log(freq)``

    Parameters
    ----------
    frequency : int
        Raw count of times the term appears in the document.
        Must be ≥ 1.

    Returns
    -------
    float
        Logarithmically scaled term frequency.

    Raises
    ------
    ValueError
        If *frequency* ≤ 0 (log(0) is undefined).
    """
    if frequency <= 0:
        raise ValueError(
            f"Term frequency must be >= 1, got {frequency}"
        )
    return 1.0 + math.log(frequency)


def inverse_document_frequency(
    total_documents: int,
    document_frequency: int,
) -> float:
    """
    Inverse document frequency.

    ``IDF = log(N / DF)``

    A term appearing in every document has IDF = 0.
    A term appearing in few documents has high IDF.

    Parameters
    ----------
    total_documents : int
        Total number of indexed documents (N). Must be ≥ 1.
    document_frequency : int
        Number of documents containing the term (DF). Must be ≥ 1
        and ≤ *total_documents*.

    Returns
    -------
    float
        Inverse document frequency.

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
    return math.log(total_documents / document_frequency)


def tf_idf(
    frequency: int,
    total_documents: int,
    document_frequency: int,
) -> float:
    """
    Combined TF-IDF weight.

    ``TF-IDF = TF(freq) × IDF(N, DF)``

    Parameters
    ----------
    frequency : int
        Raw term frequency in the document.
    total_documents : int
        Total indexed documents.
    document_frequency : int
        Documents containing this term.

    Returns
    -------
    float
        TF-IDF weight.
    """
    return (
        term_frequency(frequency)
        * inverse_document_frequency(total_documents, document_frequency)
    )


def cosine_similarity(
    query_vector: dict[str, float],
    document_vector: dict[str, float],
) -> float:
    """
    Cosine similarity between two sparse vectors.

    ``cos(Q,D) = (Q·D) / (|Q| × |D|)``

    Vectors are represented as ``{term: weight}`` dicts.
    Terms not present in a vector have an implicit weight of 0.

    Parameters
    ----------
    query_vector : dict[str, float]
        Query TF-IDF vector.
    document_vector : dict[str, float]
        Document TF-IDF vector.

    Returns
    -------
    float
        Similarity in [0, 1].  Returns 0.0 if either vector
        has zero magnitude.
    """
    # Dot product — only terms in both vectors contribute.
    dot = 0.0
    for term, q_weight in query_vector.items():
        d_weight = document_vector.get(term, 0.0)
        dot += q_weight * d_weight

    if dot == 0.0:
        return 0.0

    # Magnitudes.
    q_mag = math.sqrt(sum(w * w for w in query_vector.values()))
    d_mag = math.sqrt(sum(w * w for w in document_vector.values()))

    if q_mag == 0.0 or d_mag == 0.0:
        return 0.0

    return dot / (q_mag * d_mag)
