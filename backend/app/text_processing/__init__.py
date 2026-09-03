"""
Text processing — tokenisation, normalisation, and stop-word filtering.

Public API
----------
tokenize(text, *, remove_stopwords=False)
    Tokenise raw text into normalised tokens.

normalize_text(text)
    Apply Unicode + case + whitespace normalisation.

term_frequency(tokens)
    Count occurrences of each token.

token_count(text)
    Number of tokens in the text.

filter_stopwords(tokens)
    Remove English stop words from a token list.

Tokenizer
    Class-based interface with configurable ``TokenizerConfig``.
"""

from app.text_processing.normalizer import normalize_text
from app.text_processing.stopwords import ENGLISH_STOP_WORDS, filter_stopwords
from app.text_processing.tokenizer import (
    Tokenizer,
    term_frequency,
    token_count,
    tokenize,
)
from app.text_processing.types import TokenizerConfig

__all__ = [
    "ENGLISH_STOP_WORDS",
    "Tokenizer",
    "TokenizerConfig",
    "filter_stopwords",
    "normalize_text",
    "term_frequency",
    "token_count",
    "tokenize",
]
