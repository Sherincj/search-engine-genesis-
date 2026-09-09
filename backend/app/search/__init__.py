"""
Search package — query parsing, candidate retrieval, TF-IDF ranking.

Public API
----------
parse_query(query)
    Normalise a raw query string into tokens.

retrieve_candidates(index, query_terms)
    Find matching documents via the inverted index.

SearchEngine(index)
    Full search pipeline: parse → retrieve → TF-IDF rank.

TF-IDF functions
    term_frequency, inverse_document_frequency, tf_idf, cosine_similarity

TF-IDF ranker
    rank_candidates, RankedCandidate
"""

from app.search.candidate_retrieval import retrieve_candidates
from app.search.query_parser import parse_query
from app.search.search_engine import SearchEngine, SearchResult
from app.search.tfidf import (
    cosine_similarity,
    inverse_document_frequency,
    term_frequency,
    tf_idf,
)
from app.search.tfidf_ranker import RankedCandidate, rank_candidates
from app.search.types import CandidateMatch, SearchHit

__all__ = [
    "CandidateMatch",
    "RankedCandidate",
    "SearchEngine",
    "SearchHit",
    "SearchResult",
    "cosine_similarity",
    "inverse_document_frequency",
    "parse_query",
    "rank_candidates",
    "retrieve_candidates",
    "term_frequency",
    "tf_idf",
]
