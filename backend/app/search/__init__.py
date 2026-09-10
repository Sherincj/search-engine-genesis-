"""
Search package — query parsing, candidate retrieval, BM25 ranking.

Public API
----------
parse_query(query)
    Normalise a raw query string into tokens.

retrieve_candidates(index, query_terms)
    Find matching documents via the inverted index.

SearchEngine(index)
    Full search pipeline: parse → retrieve → BM25 rank.

BM25 functions
    bm25_idf, bm25_term_score, bm25_score

TF-IDF functions (Phase 6, preserved)
    term_frequency, inverse_document_frequency, tf_idf, cosine_similarity
"""

from app.search.bm25 import bm25_idf, bm25_score, bm25_term_score
from app.search.bm25_ranker import RankedCandidate
from app.search.bm25_ranker import rank_candidates as bm25_rank_candidates
from app.search.candidate_retrieval import retrieve_candidates
from app.search.query_parser import parse_query
from app.search.search_engine import SearchEngine, SearchResult
from app.search.tfidf import (
    cosine_similarity,
    inverse_document_frequency,
    term_frequency,
    tf_idf,
)
from app.search.tfidf_ranker import rank_candidates as tfidf_rank_candidates
from app.search.types import CandidateMatch, SearchHit

__all__ = [
    "CandidateMatch",
    "RankedCandidate",
    "SearchEngine",
    "SearchHit",
    "SearchResult",
    "bm25_idf",
    "bm25_rank_candidates",
    "bm25_score",
    "bm25_term_score",
    "cosine_similarity",
    "inverse_document_frequency",
    "parse_query",
    "retrieve_candidates",
    "term_frequency",
    "tf_idf",
    "tfidf_rank_candidates",
]
