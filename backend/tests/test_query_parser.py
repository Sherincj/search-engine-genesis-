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


# ── Phase 11: Structured query parsing ───────────────────────

from app.search.query_parser import parse_query_structured


def test_structured_basic_terms():
    q = parse_query_structured("python development")
    assert q.terms == ("python", "development")
    assert q.phrases == ()
    assert not q.has_phrases


def test_structured_single_phrase():
    q = parse_query_structured('"python development"')
    assert q.terms == ()
    assert len(q.phrases) == 1
    assert q.phrases[0].terms == ("python", "development")
    assert q.has_phrases


def test_structured_mixed_query():
    q = parse_query_structured('python "web development"')
    assert q.terms == ("python",)
    assert len(q.phrases) == 1
    assert q.phrases[0].terms == ("web", "development")


def test_structured_multiple_phrases():
    q = parse_query_structured('"python development" "web framework"')
    assert q.terms == ()
    assert len(q.phrases) == 2
    assert q.phrases[0].terms == ("python", "development")
    assert q.phrases[1].terms == ("web", "framework")


def test_structured_empty_phrase():
    """Empty quotes should not produce a phrase."""
    q = parse_query_structured('""')
    assert q.phrases == ()
    assert q.is_empty


def test_structured_single_term_phrase():
    """Single quoted term is valid."""
    q = parse_query_structured('"python"')
    assert len(q.phrases) == 1
    assert q.phrases[0].terms == ("python",)


def test_structured_unmatched_quote_opening():
    """Unmatched opening quote — treated as normal text."""
    q = parse_query_structured('"python development')
    assert q.phrases == ()
    assert "python" in q.terms
    assert "development" in q.terms


def test_structured_unmatched_quote_trailing():
    """Unmatched trailing quote — treated as normal text."""
    q = parse_query_structured('python development"')
    assert q.phrases == ()
    assert "python" in q.terms
    assert "development" in q.terms


def test_structured_phrase_normalisation():
    """Phrase terms use the same tokenizer."""
    q = parse_query_structured('"Python, DEVELOPMENT!"')
    assert q.phrases[0].terms == ("python", "development")


def test_structured_phrase_with_hyphen():
    """Hyphens split in tokenizer, so phrase gets split terms."""
    q = parse_query_structured('"web-development"')
    assert q.phrases[0].terms == ("web", "development")


def test_structured_all_terms():
    """all_terms deduplicates across terms and phrases."""
    q = parse_query_structured('python "python development"')
    terms = q.all_terms
    assert "python" in terms
    assert "development" in terms
    # Deduplicated
    assert len(terms) == len(set(terms))


def test_structured_empty_query():
    q = parse_query_structured("")
    assert q.is_empty
    assert q.terms == ()
    assert q.phrases == ()


def test_structured_phrase_then_term():
    q = parse_query_structured('"python development" python')
    assert len(q.phrases) == 1
    assert q.phrases[0].terms == ("python", "development")
    assert "python" in q.terms

