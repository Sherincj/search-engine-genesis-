"""
HTML parser — extracts structured content from crawled pages.

Extracts:
- Page title from <title> tag
- Visible body text (excluding script, style, noscript, template)
- Links from <a href=...> tags
- Canonical URL from <link rel="canonical">

Does NOT:
- Execute JavaScript
- Download external resources
- Interpret CSS
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment

from app.crawler.models import ParsedPage

# Tags whose content should never appear in the extracted body text.
_EXCLUDED_TAGS = frozenset({
    "script", "style", "noscript", "template",
    "header", "footer", "nav", "iframe",
})

# Whitespace normalization regex.
_WHITESPACE_RE = re.compile(r"\s+")


def parse_html(html: str, base_url: str | None = None) -> ParsedPage:
    """
    Parse an HTML document into structured content.

    Parameters
    ----------
    html : str
        Raw HTML string.
    base_url : str | None
        The URL the page was fetched from (for reference).

    Returns
    -------
    ParsedPage
        Extracted title, text, links, and canonical URL.
    """
    soup = BeautifulSoup(html, "html.parser")

    title = _extract_title(soup)
    text = _extract_text(soup)
    links = _extract_links(soup)
    canonical = _extract_canonical(soup)

    return ParsedPage(
        title=title,
        text=text,
        links=links,
        canonical_url=canonical,
    )


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Extract the page title from <title>."""
    tag = soup.find("title")
    if tag and tag.string:
        text = tag.string.strip()
        return text if text else None
    return None


def _extract_text(soup: BeautifulSoup) -> str:
    """
    Extract visible text from the page body.

    Removes content from script, style, noscript, template, nav,
    header, footer, and iframe tags.  Normalizes whitespace.
    """
    # Remove excluded tags entirely.
    for tag in soup.find_all(_EXCLUDED_TAGS):
        tag.decompose()

    # Remove HTML comments.
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Get text.
    raw_text = soup.get_text(separator=" ")

    # Normalize whitespace.
    text = _WHITESPACE_RE.sub(" ", raw_text).strip()

    return text


def _extract_links(soup: BeautifulSoup) -> list[str]:
    """
    Extract all <a href=...> links from the page.

    Returns raw href values (not normalized).
    Filters out empty hrefs and javascript: schemes.
    Deduplicates within the page.
    """
    seen: set[str] = set()
    links: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href:
            continue
        # Skip javascript: and other non-HTTP schemes.
        lower = href.lower()
        if lower.startswith(("javascript:", "mailto:", "tel:", "data:")):
            continue
        if href not in seen:
            seen.add(href)
            links.append(href)

    return links


def _extract_canonical(soup: BeautifulSoup) -> str | None:
    """Extract <link rel="canonical" href="...">."""
    tag = soup.find("link", rel="canonical")
    if tag and tag.get("href"):
        href = tag["href"].strip()
        return href if href else None
    return None
