"""
Tests for the query parser.

The query parser is a thin wrapper over the Phase 3 tokenizer.
These tests verify that query normalisation is consistent with
document tokenisation.
"""

from app.search.query_parser import parse_query


# ── Basic queries ────────────────────────────────────────────

def test_basic_query():
    assert parse_query("python") == ["python"]


def test_multi_word_query():
    assert parse_query("Python Programming") == ["python", "programming"]


# ── Same normalisation as documents ──────────────────────────

def test_uppercase():
    assert parse_query("PYTHON WEB") == ["python", "web"]


def test_mixed_case():
    assert parse_query("PyThOn") == ["python"]


def test_punctuation():
    assert parse_query("hello, world!") == ["hello", "world"]


def test_hyphen_splits():
    assert parse_query("web-development") == ["web", "development"]


def test_apostrophe_stripped():
    assert parse_query("don't") == ["dont"]


def test_numbers():
    assert parse_query("Python 3.12") == ["python", "3", "12"]


def test_alphanumeric_kept():
    assert parse_query("python3") == ["python3"]


def test_unicode():
    assert parse_query("café naïve") == ["café", "naïve"]


def test_unicode_chinese():
    assert parse_query("中文 搜索") == ["中文", "搜索"]


# ── Whitespace handling ──────────────────────────────────────

def test_extra_spaces():
    assert parse_query("  python    web  ") == ["python", "web"]


def test_tabs_newlines():
    assert parse_query("python\tweb\napi") == ["python", "web", "api"]


# ── Empty / whitespace-only queries ──────────────────────────

def test_empty_query():
    assert parse_query("") == []


def test_whitespace_only():
    assert parse_query("   ") == []


def test_punctuation_only():
    assert parse_query("!!! ???") == []


# ── Duplicate terms ──────────────────────────────────────────

def test_duplicate_terms_preserved():
    """The parser preserves duplicates; dedup happens in retrieval."""
    assert parse_query("python python") == ["python", "python"]
