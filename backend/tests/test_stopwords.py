"""
Tests for stop-word filtering.

Covers:
- Default behaviour (no removal)
- Removal when enabled
- Stop-word list contents
- Empty input
- All-stopwords input
- Custom stop-word set
"""

from app.text_processing import (
    ENGLISH_STOP_WORDS,
    filter_stopwords,
    tokenize,
)


# ── Default: stop words preserved ────────────────────────────

def test_default_preserves_stopwords():
    tokens = tokenize("python is a language")
    assert tokens == ["python", "is", "a", "language"]


# ── Removal when enabled ────────────────────────────────────

def test_removal_filters_stopwords():
    tokens = tokenize("python is a language", remove_stopwords=True)
    assert tokens == ["python", "language"]


def test_removal_multiple_stopwords():
    tokens = tokenize("the quick and the dead", remove_stopwords=True)
    assert tokens == ["quick", "dead"]


# ── Stop-word list contents ──────────────────────────────────

def test_common_words_in_list():
    """Verify the usual suspects are in the set."""
    expected = {"the", "a", "an", "is", "of", "to", "in", "for", "on", "and", "or"}
    assert expected.issubset(ENGLISH_STOP_WORDS)


def test_content_words_not_in_list():
    """Content words must NOT be stop words."""
    non_stopwords = {"python", "programming", "search", "engine", "web"}
    assert non_stopwords.isdisjoint(ENGLISH_STOP_WORDS)


# ── filter_stopwords directly ────────────────────────────────

def test_filter_stopwords_basic():
    result = filter_stopwords(["python", "is", "a", "language"])
    assert result == ["python", "language"]


def test_filter_stopwords_preserves_order():
    result = filter_stopwords(["a", "python", "is", "great"])
    assert result == ["python", "great"]


# ── Empty input ──────────────────────────────────────────────

def test_filter_empty_list():
    assert filter_stopwords([]) == []


def test_tokenize_empty_with_stopwords():
    assert tokenize("", remove_stopwords=True) == []


# ── All stopwords ────────────────────────────────────────────

def test_all_stopwords_removed():
    tokens = tokenize("the a an is of to in", remove_stopwords=True)
    assert tokens == []


# ── Custom stop-word set ────────────────────────────────────

def test_custom_stopwords():
    custom = frozenset({"python", "java"})
    result = filter_stopwords(["python", "is", "better", "than", "java"], custom)
    assert result == ["is", "better", "than"]


# ── Queries that depend on stop words ────────────────────────

def test_the_who_preserved():
    """'The Who' is meaningful — default tokeniser keeps stop words."""
    tokens = tokenize("The Who")
    assert tokens == ["the", "who"]


def test_to_be_or_not_preserved():
    tokens = tokenize("to be or not to be")
    assert tokens == ["to", "be", "or", "not", "to", "be"]
