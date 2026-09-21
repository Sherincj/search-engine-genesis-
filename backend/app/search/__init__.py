"""
Search package — query parsing, candidate retrieval, BM25 ranking.

Public API
----------
parse_query(query)
    Normalise a raw query string into tokens.

parse_query_structured(query)
    Parse into structured ParsedQuery with terms and phrases.

retrieve_candidates(index, query_terms)
    Find matching documents via the inverted index.

SearchEngine(index)
    Full search pipeline: parse → retrieve → phrase match → BM25 rank.

BM25 functions
    bm25_idf, bm25_term_score, bm25_score

Phrase matching
    find_phrase_positions, find_phrase_matches, PhraseMatch

TF-IDF functions (Phase 6, preserved)
    term_frequency, inverse_document_frequency, tf_idf, cosine_similarity
"""

from app.search.bm25 import bm25_idf, bm25_score, bm25_term_score
from app.search.bm25_ranker import RankedCandidate
from app.search.bm25_ranker import rank_candidates as bm25_rank_candidates
from app.search.candidate_retrieval import retrieve_candidates
from app.search.phrase_matching import (
    PhraseMatch,
    find_phrase_matches,
    find_phrase_positions,
    get_phrase_candidates,
)
from app.search.query_parser import (
    ParsedQuery,
    QueryPhrase,
    parse_query,
    parse_query_structured,
)
from app.search.search_engine import SearchEngine, SearchResult
from app.search.snippet_generator import SnippetGenerator, generate_snippet
from app.search.snippets import Snippet, SnippetPart
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
    "ParsedQuery",
    "PhraseMatch",
    "QueryPhrase",
    "RankedCandidate",
    "SearchEngine",
    "SearchHit",
    "SearchResult",
    "Snippet",
    "SnippetGenerator",
    "SnippetPart",
    "bm25_idf",
    "bm25_rank_candidates",
    "bm25_score",
    "bm25_term_score",
    "cosine_similarity",
    "find_phrase_matches",
    "find_phrase_positions",
    "generate_snippet",
    "get_phrase_candidates",
    "inverse_document_frequency",
    "parse_query",
    "parse_query_structured",
    "retrieve_candidates",
    "term_frequency",
    "tf_idf",
    "tfidf_rank_candidates",
]

