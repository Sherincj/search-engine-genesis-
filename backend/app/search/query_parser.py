"""
Query parser — converts raw search queries into normalised tokens.

Reuses the Phase 3 tokenizer so that queries and documents go through
the **exact same normalisation pipeline**.  This is critical for
search quality::

    Document text  →  Tokenizer  →  index terms
    Search query   →  Tokenizer  →  query terms
                                       ↓
                                   matching works because both sides
                                   produce identical normalised tokens
"""

from app.text_processing import tokenize


def parse_query(query: str) -> list[str]:
    """
    Parse a raw search query into normalised tokens.

    Uses the Phase 3 tokenizer with default config (lowercase,
    Unicode normalisation, **no** stop-word removal).

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    list[str]
        Normalised query tokens.  Duplicates are preserved
        (but candidate retrieval deduplicates them).

    Examples
    --------
    >>> parse_query("Python WEB-development")
    ['python', 'web', 'development']
    >>> parse_query("")
    []
    """
    return tokenize(query)
