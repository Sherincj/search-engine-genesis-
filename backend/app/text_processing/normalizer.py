"""
Text normalisation utilities.

Normalisation is the first stage of the tokenisation pipeline::

    Raw text → **normalise** → tokenise → optional stop-word filter → tokens

Keeping normalisation separate lets it be tested and evolved independently
of the token-boundary logic.
"""

import re
import unicodedata

# Pre-compiled pattern for collapsing whitespace.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Apply the full normalisation pipeline to *text*.

    Steps (in order):

    1. **Unicode NFC** — compose combining characters so that, e.g.,
       ``é`` (U+0065 + U+0301) becomes ``é`` (U+00E9).
    2. **Lowercase** — ``Python`` → ``python``.
    3. **Whitespace collapse** — runs of spaces / tabs / newlines become
       a single space; leading and trailing whitespace is stripped.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        The normalised string.

    Examples
    --------
    >>> normalize_text("  Python   IS\\nGREAT  ")
    'python is great'
    >>> normalize_text("café")
    'café'
    """
    # Step 1 — Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)

    # Step 2 — lowercase
    text = text.lower()

    # Step 3 — collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
