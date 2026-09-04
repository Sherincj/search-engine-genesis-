"""
Tests for the DocumentIndexer.

Covers:
- Index a document (tokenises + indexes)
- IndexingResult metadata
- Remove a document
- Reindex / update a document
- Multiple documents
- Bulk index
- Integration with InvertedIndex
"""

from app.indexing.document_indexer import DocumentIndexer
from app.indexing.inverted_index import InvertedIndex
from app.text_processing import tokenize


# ── helpers ──────────────────────────────────────────────────

def _make_indexer() -> DocumentIndexer:
    return DocumentIndexer(InvertedIndex())


# ── Index a document ─────────────────────────────────────────

def test_index_document():
    indexer = _make_indexer()
    result = indexer.index_document(1, "Python is a programming language")

    assert result.document_id == 1
    assert result.token_count == 5
    assert result.unique_terms == 5
    assert result.duration_ms >= 0


def test_index_document_with_duplicates():
    indexer = _make_indexer()
    result = indexer.index_document(1, "Python Python web Python")

    assert result.token_count == 4
    assert result.unique_terms == 2  # python, web


def test_index_populates_index():
    indexer = _make_indexer()
    indexer.index_document(1, "Python is great")

    index = indexer.index
    assert index.get_total_documents() == 1
    assert index.contains_term("python")
    assert index.contains_term("great")


def test_index_uses_tokenizer():
    """Verify the indexer uses the Phase 3 tokenizer (not raw text)."""
    indexer = _make_indexer()
    indexer.index_document(1, "Python, PYTHON! Web-development.")

    index = indexer.index

    # Tokenizer normalises: python (×2), web, development
    postings = index.get_postings("python")
    assert len(postings) == 1
    assert postings[0].term_frequency == 2

    # Hyphen splits
    assert index.contains_term("web")
    assert index.contains_term("development")

    # Original mixed case should not exist
    assert not index.contains_term("Python")
    assert not index.contains_term("PYTHON")


def test_index_positions_correct():
    indexer = _make_indexer()
    # "Python is popular for web development"
    # tokens: python is popular for web development
    # pos:    0      1  2       3   4   5
    indexer.index_document(2, "Python is popular for web development")

    index = indexer.index
    p = index.get_posting("python", 2)
    assert p is not None
    assert p.positions == [0]

    p_web = index.get_posting("web", 2)
    assert p_web is not None
    assert p_web.positions == [4]


# ── Remove a document ───────────────────────────────────────

def test_remove_document():
    indexer = _make_indexer()
    indexer.index_document(1, "Python web")

    removed = indexer.remove_document(1)
    assert removed is True
    assert indexer.index.get_total_documents() == 0


def test_remove_nonexistent():
    indexer = _make_indexer()
    assert indexer.remove_document(999) is False


# ── Reindex / update ────────────────────────────────────────

def test_reindex_document():
    indexer = _make_indexer()

    # First indexing
    indexer.index_document(1, "Python web framework")

    # Update — different text
    indexer.index_document(1, "Python backend development")

    index = indexer.index
    assert index.get_total_documents() == 1

    # Old term "framework" should be gone
    assert not index.contains_term("framework")
    assert not index.contains_term("web")

    # New terms present
    assert index.contains_term("backend")
    assert index.contains_term("development")
    assert index.contains_term("python")


# ── Multiple documents ──────────────────────────────────────

def test_multiple_documents():
    indexer = _make_indexer()
    indexer.index_document(1, "Python is a programming language")
    indexer.index_document(2, "Python is popular for web development")
    indexer.index_document(3, "JavaScript is popular for web development")

    index = indexer.index
    assert index.get_total_documents() == 3

    # python in docs 1, 2
    python_docs = [p.document_id for p in index.get_postings("python")]
    assert python_docs == [1, 2]

    # web in docs 2, 3
    web_docs = [p.document_id for p in index.get_postings("web")]
    assert web_docs == [2, 3]

    assert index.validate() == []


# ── Bulk index ──────────────────────────────────────────────

def test_bulk_index():
    indexer = _make_indexer()
    results = indexer.bulk_index([
        (1, "Python web"),
        (2, "JavaScript API"),
        (3, "Python API framework"),
    ])

    assert len(results) == 3
    assert results[0].document_id == 1
    assert results[1].document_id == 2
    assert results[2].document_id == 3

    assert indexer.index.get_total_documents() == 3


def test_bulk_index_empty():
    indexer = _make_indexer()
    results = indexer.bulk_index([])
    assert results == []
    assert indexer.index.get_total_documents() == 0


# ── Document statistics after indexing ───────────────────────

def test_document_length_matches_tokenizer():
    """Document length should match the tokenizer's output length."""
    indexer = _make_indexer()
    text = "Python, PYTHON! Web-development."
    indexer.index_document(1, text)

    expected_tokens = tokenize(text)
    assert indexer.index.get_document_length(1) == len(expected_tokens)


def test_average_document_length():
    indexer = _make_indexer()
    indexer.index_document(1, "a b c")       # 3 tokens
    indexer.index_document(2, "x y")          # 2 tokens

    assert indexer.index.get_average_document_length() == 2.5
