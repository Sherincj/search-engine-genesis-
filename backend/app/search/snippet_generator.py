"""
Snippet generator — extracts relevant, highlighted text from documents.

Phase 9: produces search-result snippets so users see the most
relevant portion of each matching document with query terms highlighted.

The generator is **independent** of the ranking algorithm, FastAPI,
SQLAlchemy, and the inverted index.  It operates entirely on strings
and normalised query terms.

Architecture::

    SnippetGenerator
        ├── tokenize (Phase 3)
        ├── match term positions in raw text
        ├── choose best window (prefer dense matches)
        ├── build highlighted SnippetPart list
        └── return Snippet

Design goals:
    1. Deterministic
    2. Fast (one pass over document text)
    3. Safe (HTML-escaped output)
    4. Handles empty / missing fields gracefully
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.search.snippets import Snippet, SnippetPart
from app.text_processing import tokenize

# Default snippet configuration.
DEFAULT_SNIPPET_LENGTH = 160
DEFAULT_MIN_SNIPPET_LENGTH = 80


@dataclass
class _TermOccurrence:
    """A matched term in the raw text with its character span."""

    term: str  # normalised
    start: int  # char offset in raw text
    end: int  # char offset (exclusive)


class SnippetGenerator:
    """
    Generates search-result snippets from document text.

    Parameters
    ----------
    max_length : int
        Maximum snippet length in characters.
    min_length : int
        Minimum snippet length (used when expanding windows).

    Usage::

        gen = SnippetGenerator()
        snippet = gen.generate(
            title="Python Guide",
            body="Python is a programming language...",
            query_terms=["python"],
        )
    """

    def __init__(
        self,
        max_length: int = DEFAULT_SNIPPET_LENGTH,
        min_length: int = DEFAULT_MIN_SNIPPET_LENGTH,
    ) -> None:
        self._max_length = max_length
        self._min_length = min_length
        # Regex to find word tokens in raw text (mirrors the tokenizer regex).
        self._word_re = re.compile(r"[^\W_]+", re.UNICODE)

    def generate(
        self,
        title: str | None,
        body: str | None,
        query_terms: list[str],
    ) -> Snippet | None:
        """
        Generate a snippet for a single document.

        Parameters
        ----------
        title : str | None
            Raw document title.
        body : str | None
            Raw document body text.
        query_terms : list[str]
            Normalised query terms (from the query parser).

        Returns
        -------
        Snippet | None
            A highlighted snippet, or ``None`` if no useful text exists.
        """
        if not query_terms:
            return None

        # Deduplicate query terms.
        unique_terms = set(query_terms)

        # ── Try body with matches ────────────────────────────
        if body and body.strip():
            occurrences = self._find_occurrences(body, unique_terms)
            if occurrences:
                return self._build_snippet(
                    body, occurrences, unique_terms, field="body"
                )

        # ── Try title with matches ───────────────────────────
        if title and title.strip():
            occurrences = self._find_occurrences(title, unique_terms)
            if occurrences:
                return self._build_snippet(
                    title, occurrences, unique_terms, field="title"
                )

        # ── Fallback: body without matches ───────────────────
        if body and body.strip():
            return self._build_fallback(body, field="body")

        # ── Fallback: title without matches ──────────────────
        if title and title.strip():
            return self._build_fallback(title, field="title")

        return None

    # ── Internal helpers ─────────────────────────────────────

    def _find_occurrences(
        self, text: str, terms: set[str]
    ) -> list[_TermOccurrence]:
        """
        Find all query-term occurrences in *text*, preserving
        original character spans.

        Each word in *text* is found via the tokenizer regex, then
        normalised and compared against *terms*.
        """
        occurrences: list[_TermOccurrence] = []
        for match in self._word_re.finditer(text):
            word = match.group()
            normalised = word.lower()
            if normalised in terms:
                occurrences.append(
                    _TermOccurrence(
                        term=normalised,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return occurrences

    def _choose_best_window(
        self, text: str, occurrences: list[_TermOccurrence]
    ) -> tuple[int, int]:
        """
        Choose a character window that contains the densest cluster
        of matched terms.

        Returns ``(start, end)`` character offsets into *text*.
        """
        text_len = len(text)
        max_len = self._max_length

        # If the whole text fits, return it all.
        if text_len <= max_len:
            return 0, text_len

        # Sliding-window approach: find the window that covers the
        # most unique terms.  Ties broken by leftmost position.
        best_start = occurrences[0].start
        best_unique = 0
        best_total = 0

        for i, occ in enumerate(occurrences):
            # Window starts a bit before this occurrence.
            half = max_len // 2
            win_start = max(0, occ.start - half)
            win_end = min(text_len, win_start + max_len)
            # Adjust if we hit the end.
            win_start = max(0, win_end - max_len)

            # Count unique terms in this window.
            unique: set[str] = set()
            total = 0
            for o in occurrences:
                if o.start >= win_start and o.end <= win_end:
                    unique.add(o.term)
                    total += 1

            if (len(unique) > best_unique) or (
                len(unique) == best_unique and total > best_total
            ):
                best_unique = len(unique)
                best_total = total
                best_start = win_start

        win_start = best_start
        win_end = min(text_len, win_start + max_len)

        # Snap to word boundaries.
        win_start, win_end = self._snap_to_word_boundaries(
            text, win_start, win_end
        )

        return win_start, win_end

    def _snap_to_word_boundaries(
        self, text: str, start: int, end: int
    ) -> tuple[int, int]:
        """Adjust *start*/*end* to avoid cutting words."""
        text_len = len(text)

        # Expand start backward to a word boundary.
        if start > 0:
            while start > 0 and not text[start - 1].isspace():
                start += 1
                if start >= end:
                    # Couldn't find a boundary — revert.
                    start = max(0, end - self._max_length)
                    break

        # Expand end forward to a word boundary.
        if end < text_len:
            while end < text_len and not text[end].isspace():
                end += 1

        return start, end

    def _build_snippet(
        self,
        text: str,
        occurrences: list[_TermOccurrence],
        terms: set[str],
        field: str,
    ) -> Snippet:
        """
        Build a highlighted ``Snippet`` from raw text.

        Chooses the best window, then builds ``SnippetPart`` segments
        that mark matched words.
        """
        win_start, win_end = self._choose_best_window(text, occurrences)

        # Filter occurrences to those within the window.
        window_occs = [
            o for o in occurrences
            if o.start >= win_start and o.end <= win_end
        ]

        # Build parts.
        parts: list[SnippetPart] = []
        cursor = win_start

        # Add leading ellipsis if we're not at the start.
        if win_start > 0:
            parts.append(SnippetPart(text="..."))

        for occ in window_occs:
            # Plain text before this occurrence.
            if occ.start > cursor:
                parts.append(SnippetPart(text=text[cursor:occ.start]))
            # Highlighted term (original casing).
            parts.append(
                SnippetPart(text=text[occ.start:occ.end], highlight=True)
            )
            cursor = occ.end

        # Trailing plain text.
        if cursor < win_end:
            parts.append(SnippetPart(text=text[cursor:win_end]))

        # Add trailing ellipsis if we're not at the end.
        if win_end < len(text):
            parts.append(SnippetPart(text="..."))

        # Matched terms actually in the window.
        matched = sorted({o.term for o in window_occs})

        return Snippet(parts=parts, field=field, matched_terms=matched)

    def _build_fallback(self, text: str, field: str) -> Snippet:
        """
        Build an unhighlighted fallback snippet (no query matches).
        """
        text_len = len(text)
        if text_len <= self._max_length:
            return Snippet(
                parts=[SnippetPart(text=text)],
                field=field,
                matched_terms=[],
            )

        # Take the first max_length chars, snap to word boundary.
        end = self._max_length
        while end < text_len and not text[end].isspace():
            end += 1

        snippet_text = text[:end]
        parts = [SnippetPart(text=snippet_text)]
        if end < text_len:
            parts.append(SnippetPart(text="..."))

        return Snippet(parts=parts, field=field, matched_terms=[])


def generate_snippet(
    title: str | None,
    body: str | None,
    query_terms: list[str],
    *,
    max_length: int = DEFAULT_SNIPPET_LENGTH,
) -> Snippet | None:
    """
    Convenience function — generate a snippet without constructing
    a ``SnippetGenerator`` instance.
    """
    return SnippetGenerator(max_length=max_length).generate(
        title, body, query_terms
    )
