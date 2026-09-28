"""
Search quality test fixtures — shared test data for search regression tests.

Phase 12: Provides a deterministic document corpus for verifying
ranking, phrase matching, and query processing behavior.
"""

from app.indexing.field_index import FieldIndex


def build_quality_corpus() -> FieldIndex:
    """
    Build a small deterministic search quality dataset.

    Documents::

        Doc 1: title="Python Development"  body="Python development guide."
        Doc 2: title="Development Guide"   body="Development using JavaScript and Python."
        Doc 3: title="Python"              body="Python programming language."
        Doc 4: title="Web Development"     body="Web development using multiple languages."

    Use this to verify:
    - single-term search
    - multi-term search
    - repeated terms
    - phrase search
    - mixed query
    - title/body weighting
    - snippet selection
    """
    idx = FieldIndex()
    idx.add_document(
        1,
        title_tokens=["python", "development"],
        body_tokens=["python", "development", "guide"],
    )
    idx.add_document(
        2,
        title_tokens=["development", "guide"],
        body_tokens=["development", "using", "javascript", "and", "python"],
    )
    idx.add_document(
        3,
        title_tokens=["python"],
        body_tokens=["python", "programming", "language"],
    )
    idx.add_document(
        4,
        title_tokens=["web", "development"],
        body_tokens=["web", "development", "using", "multiple", "languages"],
    )
    return idx
