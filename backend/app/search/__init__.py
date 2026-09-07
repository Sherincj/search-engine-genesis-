"""
Search package — query parsing, candidate retrieval, and ranking.

Public API
----------
parse_query(query)
    Normalise a raw query string into tokens.

retrieve_candidates(index, query_terms)
    Find matching documents via the inverted index.

SearchEngine(index)
    Full search pipeline: parse → retrieve → rank.
"""

from app.search.candidate_retrieval import retrieve_candidates
from app.search.query_parser import parse_query
from app.search.search_engine import SearchEngine, SearchResult
from app.search.types import CandidateMatch, SearchHit

__all__ = [
    "CandidateMatch",
    "SearchEngine",
    "SearchHit",
    "SearchResult",
    "parse_query",
    "retrieve_candidates",
]
