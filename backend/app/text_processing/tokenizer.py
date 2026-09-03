"""
Tokenisation pipeline.

Converts raw text into a deterministic list of normalised tokens::

    Raw text
       ↓
    Unicode normalisation (NFC)
       ↓
    Case normalisation (lowercase)
       ↓
    Apostrophe stripping
       ↓
    Token boundary detection (regex: Unicode letters + digits)
       ↓
    Optional stop-word filtering
       ↓
    Tokens  →  list[str]

Design choices (V1)
-------------------
* **Hyphens** split tokens: ``web-development`` → ``["web", "development"]``.
* **Apostrophes** are stripped: ``don't`` → ``"dont"`` → ``["dont"]``.
  Both ASCII ``'`` (U+0027) and typographic ``'`` / ``'`` are handled.
* **Decimal dots** split tokens: ``3.14`` → ``["3", "14"]``.
  Queries go through the same pipeline, so matching still works.
* **Alphanumeric** tokens are kept whole: ``python3`` → ``["python3"]``.
* **Underscores** act as boundaries: ``my_func`` → ``["my", "func"]``.
* **Duplicate tokens are preserved** — term frequency depends on them.
* **No stemming / lemmatisation** in V1.

The same tokeniser is designed to be reused for both documents and
search queries so that index keys and query keys are always comparable.
"""

import re
import unicodedata
from collections import Counter

from app.text_processing.normalizer import normalize_text
from app.text_processing.stopwords import filter_stopwords
from app.text_processing.types import TokenizerConfig

# ---------------------------------------------------------------------------
# Regex: one or more Unicode letters / digits, excluding underscore.
#
# [^\W_]+  means "characters that are NOT (non-word OR underscore)"
#          i.e.  Unicode letters + digits only.
# ---------------------------------------------------------------------------
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

# Apostrophe / quote characters to strip before tokenisation.
_APOSTROPHES = str.maketrans("", "", "'\u2018\u2019")


# ── Tokenizer class ─────────────────────────────────────────

class Tokenizer:
    """
    Configurable tokenisation pipeline.

    Parameters
    ----------
    config : TokenizerConfig | None
        Override default behaviour.  ``None`` uses the default config
        (lowercase on, Unicode normalisation on, stop-word removal off).

    Examples
    --------
    >>> t = Tokenizer()
    >>> t.tokenize("Python is GREAT!")
    ['python', 'is', 'great']

    >>> t2 = Tokenizer(TokenizerConfig(remove_stopwords=True))
    >>> t2.tokenize("Python is a language")
    ['python', 'language']
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenise *text* into a list of normalised string tokens.

        Raises ``TypeError`` if *text* is not a ``str``.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Expected str, got {type(text).__name__}"
            )

        # Step 1 — Unicode normalisation
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Step 2 — Case normalisation
        if self.config.lowercase:
            text = text.lower()

        # Step 3 — Strip apostrophes (ASCII + typographic)
        text = text.translate(_APOSTROPHES)

        # Step 4 — Extract tokens (Unicode letters + digits)
        tokens: list[str] = _TOKEN_PATTERN.findall(text)

        # Step 5 — Optional stop-word removal
        if self.config.remove_stopwords:
            tokens = filter_stopwords(tokens)

        return tokens


# ── Module-level convenience API ─────────────────────────────

_default_tokenizer = Tokenizer()
_stopword_tokenizer = Tokenizer(TokenizerConfig(remove_stopwords=True))


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    """
    Tokenise *text* using the default configuration.

    This is the recommended entry-point for most callers.

    Parameters
    ----------
    text : str
        Raw input text.
    remove_stopwords : bool
        If ``True``, English stop words are removed from the result.

    Returns
    -------
    list[str]
        Deterministic, ordered list of normalised tokens.
        Duplicates are preserved (important for term frequency).

    Examples
    --------
    >>> tokenize("Python is GREAT!")
    ['python', 'is', 'great']

    >>> tokenize("Python is a language", remove_stopwords=True)
    ['python', 'language']
    """
    if remove_stopwords:
        return _stopword_tokenizer.tokenize(text)
    return _default_tokenizer.tokenize(text)


# ── Utilities ────────────────────────────────────────────────

def term_frequency(tokens: list[str]) -> dict[str, int]:
    """
    Count occurrences of each token.

    This utility will later feed TF-IDF and BM25.

    Parameters
    ----------
    tokens : list[str]
        Token list (may contain duplicates).

    Returns
    -------
    dict[str, int]
        Mapping of token → count.

    Examples
    --------
    >>> term_frequency(["python", "python", "web"])
    {'python': 2, 'web': 1}
    """
    return dict(Counter(tokens))


def token_count(text: str) -> int:
    """
    Return the number of tokens in *text* using the default tokeniser.

    Equivalent to ``len(tokenize(text))`` but reads a little better
    at call sites.
    """
    return len(tokenize(text))
