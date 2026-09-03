"""
Token-related data structures and configuration.

Keeps configuration separate from implementation so the tokenizer
can be reconfigured without touching logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerConfig:
    """
    Controls tokenizer behaviour.

    Attributes
    ----------
    lowercase : bool
        Normalise tokens to lowercase (default ``True``).
    normalize_unicode : bool
        Apply Unicode NFC normalisation (default ``True``).
    remove_stopwords : bool
        Strip English stop words from the token list
        (default ``False`` — stop words are preserved).
    """

    lowercase: bool = True
    normalize_unicode: bool = True
    remove_stopwords: bool = False
