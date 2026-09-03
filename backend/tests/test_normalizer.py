"""
Tests for the text normalisation module.

Covers:
- Lowercase conversion
- Unicode NFC normalisation
- Whitespace collapsing
- Empty / whitespace-only input
"""

import unicodedata

import pytest

from app.text_processing.normalizer import normalize_text


# ── Lowercase ────────────────────────────────────────────────

def test_lowercase_basic():
    assert normalize_text("Python") == "python"


def test_lowercase_all_caps():
    assert normalize_text("HELLO WORLD") == "hello world"


def test_lowercase_mixed():
    assert normalize_text("PyThOn Is GrEaT") == "python is great"


def test_already_lowercase():
    assert normalize_text("python") == "python"


# ── Unicode NFC ──────────────────────────────────────────────

def test_unicode_nfc_composed():
    """Combining characters should be composed to NFC form."""
    # é as e + combining acute accent
    decomposed = "caf\u0065\u0301"
    result = normalize_text(decomposed)
    assert result == "café"
    # Verify NFC
    assert result == unicodedata.normalize("NFC", result)


def test_unicode_already_nfc():
    assert normalize_text("café") == "café"


def test_unicode_naive():
    """'naïve' with combining diaeresis should normalise."""
    assert normalize_text("naïve") == "naïve"


# ── Whitespace ───────────────────────────────────────────────

def test_multiple_spaces():
    assert normalize_text("hello    world") == "hello world"


def test_tabs():
    assert normalize_text("hello\tworld") == "hello world"


def test_newlines():
    assert normalize_text("hello\nworld") == "hello world"


def test_mixed_whitespace():
    assert normalize_text("  hello \t world \n python  ") == "hello world python"


def test_leading_trailing_whitespace():
    assert normalize_text("  python  ") == "python"


# ── Empty / whitespace-only ──────────────────────────────────

def test_empty_string():
    assert normalize_text("") == ""


def test_spaces_only():
    assert normalize_text("   ") == ""


def test_whitespace_chars_only():
    assert normalize_text("\n\t") == ""


# ── Punctuation preservation ─────────────────────────────────
# normalize_text does NOT strip punctuation — that's the tokenizer's job.

def test_punctuation_preserved():
    """Normaliser keeps punctuation; tokeniser handles boundaries."""
    assert normalize_text("Hello, World!") == "hello, world!"
