"""
Tests for the IndexBuilder.

Covers:
- Build from individual add_document calls
- from_documents static factory
- Empty build
- Builder produces valid index
"""

from app.indexing.index_builder import IndexBuilder


def test_builder_add_and_build():
    builder = IndexBuilder()
    builder.add_document(1, ["python", "web"])
    builder.add_document(2, ["python", "api"])

    index = builder.build()

    assert index.get_total_documents() == 2
    assert index.get_document_frequency("python") == 2
    assert index.get_document_frequency("web") == 1
    assert index.get_document_frequency("api") == 1


def test_builder_from_documents():
    index = IndexBuilder.from_documents([
        (1, ["python", "web"]),
        (2, ["python", "api"]),
        (3, ["javascript", "web"]),
    ])

    assert index.get_total_documents() == 3
    assert index.get_document_frequency("python") == 2
    assert index.get_document_frequency("web") == 2
    assert index.get_document_frequency("javascript") == 1


def test_builder_empty():
    builder = IndexBuilder()
    index = builder.build()

    assert index.get_total_documents() == 0
    assert index.vocabulary_size == 0


def test_builder_produces_valid_index():
    index = IndexBuilder.from_documents([
        (1, ["python", "web", "python"]),
        (2, ["javascript", "web"]),
        (3, ["python", "api", "python", "python"]),
    ])

    assert index.validate() == []


def test_builder_preserves_positions():
    index = IndexBuilder.from_documents([
        (1, ["python", "web", "python"]),
    ])

    postings = index.get_postings("python")
    assert postings[0].term_frequency == 2
    assert postings[0].positions == [0, 2]


def test_builder_from_documents_empty_list():
    index = IndexBuilder.from_documents([])
    assert index.get_total_documents() == 0
