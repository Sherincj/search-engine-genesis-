"""
Stop-word definitions and filtering.

Stop-word removal is **optional** and **disabled by default** in the
tokeniser.  Some queries (e.g. ``"to be or not to be"``, ``"The Who"``)
depend on stop words for meaning, so the base tokeniser preserves them.

The indexing or search layer can choose to enable removal when appropriate.
"""

# A conservative English stop-word set.
# Kept deliberately small — aggressive removal hurts recall.
ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
)


def filter_stopwords(
    tokens: list[str],
    stop_words: frozenset[str] | None = None,
) -> list[str]:
    """
    Remove stop words from a token list.

    Parameters
    ----------
    tokens : list[str]
        Already-normalised tokens (lowercase).
    stop_words : frozenset[str] | None
        Custom stop-word set.  Falls back to ``ENGLISH_STOP_WORDS``.

    Returns
    -------
    list[str]
        Tokens with stop words removed, preserving order.

    Examples
    --------
    >>> filter_stopwords(["python", "is", "a", "language"])
    ['python', 'language']
    """
    if stop_words is None:
        stop_words = ENGLISH_STOP_WORDS
    return [t for t in tokens if t not in stop_words]
