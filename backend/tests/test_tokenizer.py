"""
Tests for the tokeniser.

Covers:
- Basic text
- Uppercase / case normalisation
- Punctuation removal
- Whitespace handling
- Empty / whitespace-only input
- Unicode text (Latin, Chinese, Japanese, Korean)
- Hyphens → split
- Apostrophes → strip
- Numbers, decimals, alphanumeric
- URLs, emails (deterministic, no crash)
- None → TypeError
- Determinism
- Duplicate preservation
- Tokenizer class + config
- term_frequency utility
- token_count utility
"""

import pytest

from app.text_processing import (
    Tokenizer,
    TokenizerConfig,
    term_frequency,
    token_count,
    tokenize,
)


# ── Basic text ───────────────────────────────────────────────

def test_basic_text():
    assert tokenize("Python is great") == ["python", "is", "great"]


def test_sentence():
    assert tokenize("Python is a programming language") == [
        "python", "is", "a", "programming", "language",
    ]


# ── Uppercase / case normalisation ───────────────────────────

def test_all_uppercase():
    assert tokenize("PYTHON PYTHON PYTHON") == ["python", "python", "python"]


def test_mixed_case():
    assert tokenize("PyThOn pYtHoN") == ["python", "python"]


def test_case_preserved_when_disabled():
    t = Tokenizer(TokenizerConfig(lowercase=False))
    assert t.tokenize("Python PYTHON") == ["Python", "PYTHON"]


# ── Punctuation ──────────────────────────────────────────────

def test_comma():
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_multiple_punctuation():
    assert tokenize("Wow!!! Really???") == ["wow", "really"]


def test_all_punctuation_types():
    text = 'a. b, c! d? e: f; g( h) i[ j] k{ l} m" n'
    tokens = tokenize(text)
    assert tokens == list("abcdefghijklmn")


def test_surrounding_punctuation():
    assert tokenize("(hello) [world]") == ["hello", "world"]


# ── Whitespace ───────────────────────────────────────────────

def test_multiple_spaces():
    assert tokenize("hello    world") == ["hello", "world"]


def test_tabs():
    assert tokenize("hello\tworld") == ["hello", "world"]


def test_newlines():
    assert tokenize("hello\nworld\npython") == ["hello", "world", "python"]


def test_mixed_whitespace():
    assert tokenize("hello    world\npython") == ["hello", "world", "python"]


# ── Empty input ──────────────────────────────────────────────

def test_empty_string():
    assert tokenize("") == []


def test_whitespace_only():
    assert tokenize("   ") == []


def test_newline_tab_only():
    assert tokenize("\n\t") == []


# ── Unicode ──────────────────────────────────────────────────

def test_unicode_cafe():
    assert tokenize("café") == ["café"]


def test_unicode_naive():
    assert tokenize("naïve approach") == ["naïve", "approach"]


def test_unicode_chinese():
    assert tokenize("中文") == ["中文"]


def test_unicode_japanese():
    tokens = tokenize("日本語テスト")
    assert tokens == ["日本語テスト"]


def test_unicode_korean():
    assert tokenize("한국어") == ["한국어"]


def test_unicode_accented():
    assert tokenize("résumé über") == ["résumé", "über"]


# ── Hyphens → split ─────────────────────────────────────────

def test_hyphen_simple():
    assert tokenize("web-development") == ["web", "development"]


def test_hyphen_multiple():
    assert tokenize("state-of-the-art") == ["state", "of", "the", "art"]


def test_hyphen_with_numbers():
    assert tokenize("Python-3") == ["python", "3"]


# ── Apostrophes → strip ─────────────────────────────────────

def test_apostrophe_contraction():
    assert tokenize("don't") == ["dont"]


def test_apostrophe_possessive():
    assert tokenize("user's code") == ["users", "code"]


def test_apostrophe_trailing():
    assert tokenize("James'") == ["james"]


def test_unicode_right_quote():
    """Typographic right single quote (U+2019) is stripped too."""
    assert tokenize("don\u2019t") == ["dont"]


def test_unicode_left_quote():
    """Typographic left single quote (U+2018) is stripped too."""
    assert tokenize("\u2018hello\u2019") == ["hello"]


# ── Numbers ──────────────────────────────────────────────────

def test_integer():
    assert tokenize("2026") == ["2026"]


def test_decimal_splits():
    """Decimal dot acts as a boundary: 3.14 → ['3', '14']."""
    assert tokenize("3.14") == ["3", "14"]


def test_number_in_context():
    assert tokenize("Python 3.12") == ["python", "3", "12"]


def test_large_number():
    assert tokenize("RTX 5060") == ["rtx", "5060"]


# ── Alphanumeric (kept whole) ────────────────────────────────

def test_alphanumeric_python3():
    assert tokenize("python3") == ["python3"]


def test_alphanumeric_web3():
    assert tokenize("web3") == ["web3"]


def test_alphanumeric_rtx5060():
    assert tokenize("rtx5060") == ["rtx5060"]


def test_alphanumeric_windows11():
    assert tokenize("Windows11 is here") == ["windows11", "is", "here"]


# ── Underscores (act as boundaries) ─────────────────────────

def test_underscore():
    assert tokenize("my_function") == ["my", "function"]


# ── URLs in text (no crash, deterministic) ───────────────────

def test_url_in_text():
    tokens = tokenize("visit https://example.com/python today")
    assert "visit" in tokens
    assert "https" in tokens
    assert "example" in tokens
    assert "com" in tokens
    assert "python" in tokens
    assert "today" in tokens


# ── Emails in text (no crash, deterministic) ─────────────────

def test_email_in_text():
    tokens = tokenize("contact user@example.com please")
    assert "contact" in tokens
    assert "user" in tokens
    assert "example" in tokens
    assert "com" in tokens
    assert "please" in tokens


# ── None → TypeError ────────────────────────────────────────

def test_none_raises_type_error():
    with pytest.raises(TypeError, match="Expected str, got NoneType"):
        tokenize(None)  # type: ignore[arg-type]


def test_int_raises_type_error():
    with pytest.raises(TypeError, match="Expected str, got int"):
        tokenize(42)  # type: ignore[arg-type]


# ── Determinism ──────────────────────────────────────────────

def test_determinism():
    text = "Python is a GREAT programming language! café naïve web-dev"
    assert tokenize(text) == tokenize(text)


def test_determinism_repeated():
    text = "RTX 5060 benchmarks"
    results = [tokenize(text) for _ in range(100)]
    assert all(r == results[0] for r in results)


# ── Duplicate preservation ───────────────────────────────────

def test_duplicates_preserved():
    assert tokenize("python python python web") == [
        "python", "python", "python", "web",
    ]


def test_duplicates_from_mixed_case():
    assert tokenize("Python PYTHON python") == [
        "python", "python", "python",
    ]


# ── Tokenizer class with config ─────────────────────────────

def test_tokenizer_default_config():
    t = Tokenizer()
    assert t.tokenize("Hello World") == ["hello", "world"]


def test_tokenizer_no_lowercase():
    t = Tokenizer(TokenizerConfig(lowercase=False))
    assert t.tokenize("Hello World") == ["Hello", "World"]


def test_tokenizer_no_unicode_normalization():
    t = Tokenizer(TokenizerConfig(normalize_unicode=False))
    # Should still work, just skips NFC step
    assert t.tokenize("hello world") == ["hello", "world"]


def test_tokenizer_with_stopwords():
    t = Tokenizer(TokenizerConfig(remove_stopwords=True))
    assert t.tokenize("Python is a language") == ["python", "language"]


# ── Convenience function with stopwords ──────────────────────

def test_tokenize_with_stopwords():
    result = tokenize("Python is a great language", remove_stopwords=True)
    assert result == ["python", "great", "language"]


def test_tokenize_without_stopwords():
    result = tokenize("Python is a great language", remove_stopwords=False)
    assert result == ["python", "is", "a", "great", "language"]


# ── term_frequency ───────────────────────────────────────────

def test_term_frequency_basic():
    freq = term_frequency(["python", "python", "web"])
    assert freq == {"python": 2, "web": 1}


def test_term_frequency_single():
    freq = term_frequency(["hello"])
    assert freq == {"hello": 1}


def test_term_frequency_empty():
    freq = term_frequency([])
    assert freq == {}


def test_term_frequency_all_unique():
    freq = term_frequency(["a", "b", "c"])
    assert freq == {"a": 1, "b": 1, "c": 1}


def test_term_frequency_from_tokenize():
    tokens = tokenize("Python Python web Python web")
    freq = term_frequency(tokens)
    assert freq == {"python": 3, "web": 2}


# ── token_count ──────────────────────────────────────────────

def test_token_count_basic():
    assert token_count("Python is great") == 3


def test_token_count_empty():
    assert token_count("") == 0


def test_token_count_punctuation():
    # "hello,world" → ["hello", "world"] → 2
    assert token_count("hello,world") == 2


# ── Mixed / realistic content ────────────────────────────────

def test_realistic_paragraph():
    text = (
        "Python is a high-level, general-purpose programming language. "
        "Its design philosophy emphasizes code readability. Python 3.12 "
        "was released in 2023."
    )
    tokens = tokenize(text)
    assert "python" in tokens
    assert "high" in tokens
    assert "level" in tokens
    assert "3" in tokens
    assert "12" in tokens
    assert "2023" in tokens
    # Duplicates preserved
    assert tokens.count("python") == 2
