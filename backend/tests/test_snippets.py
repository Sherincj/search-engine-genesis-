"""
Tests for the snippet generator.

Covers:
- Basic snippet generation
- Empty / None inputs
- Case-insensitive matching
- Multiple terms
- Duplicate query terms
- Unknown terms
- Punctuation handling
- Unicode support
- Long document windowing
- Match-centered extraction
- Word boundary snapping
- Ellipsis behavior
- Dense match preference
- Field selection (title vs body)
- HTML safety / XSS prevention
- Snippet data structure
- Fallback behavior
"""

import html as html_mod

from app.search.snippet_generator import SnippetGenerator, generate_snippet
from app.search.snippets import Snippet, SnippetPart


# ── Helpers ──────────────────────────────────────────────────

def _gen(
    body: str | None,
    terms: list[str],
    *,
    title: str | None = None,
    max_length: int = 160,
) -> Snippet | None:
    """Helper to generate a snippet directly."""
    return SnippetGenerator(max_length=max_length).generate(
        title=title, body=body, query=terms
    )


# ── Basic ────────────────────────────────────────────────────

def test_generates_snippet():
    s = _gen(body="Python is a programming language", terms=["python"])
    assert s is not None
    assert s.field == "body"
    assert "python" in s.matched_terms


def test_returns_original_text_when_short():
    s = _gen(body="Python web", terms=["python"])
    assert s is not None
    assert "Python" in s.plain_text
    # No ellipsis when text is short.
    assert "..." not in s.plain_text


def test_handles_empty_text():
    s = _gen(body="", terms=["python"])
    assert s is None


def test_handles_none_body():
    s = _gen(body=None, title=None, terms=["python"])
    assert s is None


def test_handles_empty_query():
    s = _gen(body="Python is great", terms=[])
    assert s is None


def test_convenience_function():
    s = generate_snippet("Title", "Body with python", ["python"])
    assert s is not None
    assert "python" in s.matched_terms


# ── Matching ─────────────────────────────────────────────────

def test_case_insensitive_matching():
    s = _gen(body="Python is GREAT for web DEVELOPMENT", terms=["python", "development"])
    assert s is not None
    assert "python" in s.matched_terms
    assert "development" in s.matched_terms
    # Original casing preserved in output.
    assert "Python" in s.plain_text
    assert "DEVELOPMENT" in s.plain_text


def test_multiple_terms():
    s = _gen(body="Python is great for web development", terms=["python", "web"])
    assert s is not None
    assert "python" in s.matched_terms
    assert "web" in s.matched_terms


def test_duplicate_query_terms():
    s1 = _gen(body="Python is great", terms=["python"])
    s2 = _gen(body="Python is great", terms=["python", "python", "python"])
    assert s1 is not None and s2 is not None
    assert s1.plain_text == s2.plain_text
    assert s1.matched_terms == s2.matched_terms


def test_unknown_terms_fallback():
    """Unknown terms produce a fallback snippet (no highlights)."""
    s = _gen(body="Python is a language", terms=["javascript"])
    assert s is not None
    assert s.matched_terms == []
    # Fallback uses body text.
    assert "Python" in s.plain_text


def test_punctuation_in_text():
    s = _gen(body="Python, the language! Is great.", terms=["python"])
    assert s is not None
    assert "python" in s.matched_terms
    # Punctuation preserved in output.
    assert "Python," in s.plain_text


def test_numbers():
    s = _gen(body="Python 3 is the latest version", terms=["3"])
    assert s is not None
    assert "3" in s.matched_terms


# ── Unicode ──────────────────────────────────────────────────

def test_unicode_cafe():
    s = _gen(body="Le café est excellent", terms=["café"])
    assert s is not None
    assert "café" in s.matched_terms


def test_unicode_chinese():
    s = _gen(body="Python 是一种编程语言", terms=["python"])
    assert s is not None
    assert "python" in s.matched_terms


def test_unicode_korean():
    s = _gen(body="한국어 프로그래밍", terms=["한국어"])
    assert s is not None
    assert "한국어" in s.matched_terms


# ── Windowing ────────────────────────────────────────────────

def test_long_document_truncated():
    body = "filler " * 200 + "Python is here" + " filler" * 200
    s = _gen(body=body, terms=["python"], max_length=80)
    assert s is not None
    # Should be much shorter than the full text.
    assert len(s.plain_text) < len(body)


def test_matching_region_preferred():
    body = "A " * 100 + "Python is excellent" + " B" * 100
    s = _gen(body=body, terms=["python"], max_length=80)
    assert s is not None
    assert "Python" in s.plain_text


def test_ellipsis_when_truncated():
    body = "Start " + "filler " * 50 + "Python is great" + " filler" * 50
    s = _gen(body=body, terms=["python"], max_length=80)
    assert s is not None
    assert "..." in s.plain_text


def test_no_ellipsis_when_fits():
    s = _gen(body="Python is great", terms=["python"], max_length=200)
    assert s is not None
    assert "..." not in s.plain_text


# ── Dense match preference ───────────────────────────────────

def test_nearby_terms_preferred():
    """
    When 'python' and 'development' appear close together later in
    the document, that region should be chosen over an isolated match.
    """
    body = (
        "Python is a language. " +
        "Other unrelated text. " * 20 +
        "Many developers use Python for web development because it is simple."
    )
    s = _gen(body=body, terms=["python", "development"], max_length=120)
    assert s is not None
    # Both terms should be in the snippet.
    assert "python" in s.matched_terms
    assert "development" in s.matched_terms


# ── Fields ───────────────────────────────────────────────────

def test_body_match_over_title():
    """Body with matches is preferred over title with matches."""
    s = _gen(
        title="Python Guide",
        body="Learn Python programming today",
        terms=["python"],
    )
    assert s is not None
    assert s.field == "body"


def test_title_match_when_body_has_none():
    s = _gen(
        title="Python Programming Guide",
        body="This guide explains concepts",
        terms=["python"],
    )
    assert s is not None
    assert s.field == "title"
    assert "Python" in s.plain_text


def test_title_only_document():
    s = _gen(title="Python Guide", body=None, terms=["python"])
    assert s is not None
    assert s.field == "title"


def test_body_only_document():
    s = _gen(title=None, body="Python is great", terms=["python"])
    assert s is not None
    assert s.field == "body"


def test_title_and_body_match():
    """When both fields match, body is preferred."""
    s = _gen(
        title="Python Tutorial",
        body="Python web development guide",
        terms=["python"],
    )
    assert s is not None
    assert s.field == "body"


def test_no_match_body_fallback():
    """When query doesn't match body, produce fallback from body."""
    s = _gen(
        title="Guide",
        body="This document contains information about programming",
        terms=["javascript"],
    )
    assert s is not None
    assert s.field == "body"
    assert s.matched_terms == []


def test_no_match_title_fallback():
    """When body is empty and query doesn't match title."""
    s = _gen(title="Programming Guide", body="", terms=["javascript"])
    assert s is not None
    assert s.field == "title"
    assert s.matched_terms == []


def test_both_empty():
    s = _gen(title="", body="", terms=["python"])
    assert s is None


def test_whitespace_only_body():
    s = _gen(title=None, body="   \n\t  ", terms=["python"])
    assert s is None


# ── Security (HTML escaping / XSS prevention) ────────────────

def test_html_entities_escaped():
    s = _gen(body="Python & JavaScript are great", terms=["python"])
    assert s is not None
    html_output = s.to_html()
    assert "&amp;" in html_output
    assert "& " not in html_output.replace("&amp;", "")


def test_script_tag_escaped():
    """Malicious script tags must be escaped."""
    s = _gen(
        body='<script>alert("x")</script> Python is great',
        terms=["python"],
    )
    assert s is not None
    html_output = s.to_html()
    assert "<script>" not in html_output
    assert "&lt;script&gt;" in html_output


def test_img_onerror_escaped():
    s = _gen(
        body='<img src=x onerror=alert(1)> Python rocks',
        terms=["python"],
    )
    assert s is not None
    html_output = s.to_html()
    assert "onerror" not in html_output or "&lt;img" in html_output
    assert "<img" not in html_output


def test_angle_brackets_escaped():
    s = _gen(body="a < b > c Python", terms=["python"])
    assert s is not None
    html_output = s.to_html()
    assert "&lt;" in html_output
    assert "&gt;" in html_output


def test_quotes_escaped():
    s = _gen(body='She said "Python is great"', terms=["python"])
    assert s is not None
    html_output = s.to_html()
    assert "&quot;" in html_output


# ── Snippet structure ────────────────────────────────────────

def test_snippet_parts_structure():
    s = _gen(body="Python is great", terms=["python"])
    assert s is not None
    # Should have at least a highlighted part and a plain part.
    highlights = [p for p in s.parts if p.highlight]
    plains = [p for p in s.parts if not p.highlight]
    assert len(highlights) >= 1
    assert highlights[0].text == "Python"


def test_snippet_to_dict():
    s = _gen(body="Python is great", terms=["python"])
    assert s is not None
    d = s.to_dict()
    assert "snippet_html" in d
    assert "snippet_parts" in d
    assert isinstance(d["snippet_parts"], list)
    assert any(p["highlight"] for p in d["snippet_parts"])


def test_plain_text_property():
    s = _gen(body="Learn Python programming", terms=["python"])
    assert s is not None
    assert s.plain_text == "Learn Python programming"


# ── Frozen dataclass ─────────────────────────────────────────

def test_snippet_is_frozen():
    s = _gen(body="Python is great", terms=["python"])
    assert s is not None
    try:
        s.field = "other"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_snippet_part_is_frozen():
    p = SnippetPart(text="hello", highlight=True)
    try:
        p.text = "world"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
