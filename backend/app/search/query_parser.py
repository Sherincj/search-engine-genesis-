"""
Query parser — converts raw search queries into a structured representation.

Phase 11 update: supports quoted phrase syntax::

    python                  → terms: [python]
    python development      → terms: [python, development]
    "python development"    → phrases: [(python, development)]
    python "web dev"        → terms: [python], phrases: [(web, dev)]

Reuses the Phase 3 tokenizer so that queries and documents go through
the **exact same normalisation pipeline**.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.text_processing import tokenize


@dataclass(frozen=True)
class QueryPhrase:
    """A quoted phrase — terms that must appear adjacent in order."""

    terms: tuple[str, ...]


@dataclass(frozen=True)
class ParsedQuery:
    """
    Structured query representation.

    Attributes
    ----------
    terms : tuple[str, ...]
        Individual (unquoted) search terms.
    phrases : tuple[QueryPhrase, ...]
        Quoted phrases requiring exact adjacency.
    all_terms : tuple[str, ...]
        All unique terms from both individual terms and phrases,
        used for candidate retrieval.
    """

    terms: tuple[str, ...]
    phrases: tuple[QueryPhrase, ...]

    @property
    def all_terms(self) -> tuple[str, ...]:
        """All unique terms for candidate retrieval."""
        seen: set[str] = set()
        result: list[str] = []
        for t in self.terms:
            if t not in seen:
                seen.add(t)
                result.append(t)
        for phrase in self.phrases:
            for t in phrase.terms:
                if t not in seen:
                    seen.add(t)
                    result.append(t)
        return tuple(result)

    @property
    def has_phrases(self) -> bool:
        return len(self.phrases) > 0

    @property
    def is_empty(self) -> bool:
        return len(self.terms) == 0 and len(self.phrases) == 0


# Regex to extract quoted phrases: "..." or "..."
_PHRASE_RE = re.compile(r'"([^"]*)"')


def parse_query(query: str) -> list[str]:
    """
    Parse a raw search query into normalised tokens.

    Legacy interface — returns a flat list of all terms.
    Maintained for backward compatibility.  Duplicates are
    preserved (candidate retrieval deduplicates them).

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    list[str]
        Normalised query tokens.

    Examples
    --------
    >>> parse_query("Python WEB-development")
    ['python', 'web', 'development']
    >>> parse_query("")
    []
    """
    return tokenize(query)


def parse_query_structured(query: str) -> ParsedQuery:
    """
    Parse a raw search query into a structured representation
    that distinguishes individual terms from quoted phrases.

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    ParsedQuery
        Structured query with terms and phrases.

    Examples
    --------
    >>> q = parse_query_structured('"python development"')
    >>> q.phrases[0].terms
    ('python', 'development')

    >>> q = parse_query_structured('python "web development"')
    >>> q.terms
    ('python',)
    >>> q.phrases[0].terms
    ('web', 'development')
    """
    if not query or not query.strip():
        return ParsedQuery(terms=(), phrases=())

    # 1. Extract quoted phrases.
    phrases: list[QueryPhrase] = []
    remaining = query

    for match in _PHRASE_RE.finditer(query):
        phrase_text = match.group(1)
        phrase_tokens = tokenize(phrase_text)
        if len(phrase_tokens) >= 1:
            phrases.append(QueryPhrase(terms=tuple(phrase_tokens)))

    # 2. Remove quoted phrases from query to get remaining terms.
    remaining = _PHRASE_RE.sub("", remaining)

    # 3. Tokenize remaining text as individual terms.
    terms = tokenize(remaining)

    return ParsedQuery(
        terms=tuple(terms),
        phrases=tuple(phrases),
    )
