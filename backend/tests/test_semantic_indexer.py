"""Tests for the semantic indexer (incremental updates)."""

from app.indexing.semantic_indexer import SemanticIndexer
from app.vector_index.vector_index import VectorIndex


class _FakeProvider:
    def __init__(self, dim=2):
        self._dim = dim

    def embed_text(self, text: str) -> list[float]:
        # Return a vector based on text length for testing
        val = len(text) / 100.0
        return [val, val]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self):
        return self._dim

    @property
    def model_name(self):
        return "fake"


def test_index_new_document():
    idx = VectorIndex(2)
    indexer = SemanticIndexer(idx, _FakeProvider())
    
    indexer.index_document(1, "Title", "Body")
    assert idx.contains(1)
    assert idx.count() == 1


def test_index_updated_document():
    idx = VectorIndex(2)
    indexer = SemanticIndexer(idx, _FakeProvider())
    
    indexer.index_document(1, "Title", "Body")
    v1 = idx.get_vector(1)
    
    indexer.update_document(1, "Longer Title", "Much longer body text")
    v2 = idx.get_vector(1)
    
    assert idx.count() == 1
    assert v1 != v2  # Vector was replaced


def test_remove_document():
    idx = VectorIndex(2)
    indexer = SemanticIndexer(idx, _FakeProvider())
    
    indexer.index_document(1, "Title", "Body")
    assert idx.contains(1)
    
    removed = indexer.remove_document(1)
    assert removed is True
    assert not idx.contains(1)
    assert idx.count() == 0


def test_remove_nonexistent_document():
    idx = VectorIndex(2)
    indexer = SemanticIndexer(idx, _FakeProvider())
    
    removed = indexer.remove_document(99)
    assert removed is False


def test_reindex_does_not_duplicate():
    idx = VectorIndex(2)
    indexer = SemanticIndexer(idx, _FakeProvider())
    
    indexer.index_document(1, "Title", "Body")
    indexer.index_document(1, "Title", "Body")
    
    assert idx.count() == 1


def test_empty_document_ignored():
    idx = VectorIndex(2)
    indexer = SemanticIndexer(idx, _FakeProvider())
    
    indexer.index_document(1, "", "")
    assert not idx.contains(1)
    assert idx.count() == 0


def test_bulk_index():
    idx = VectorIndex(2)
    indexer = SemanticIndexer(idx, _FakeProvider())
    
    docs = [
        (1, "A", "Text A"),
        (2, "B", "Text B"),
        (3, "", ""),  # Should be ignored
    ]
    
    indexer.bulk_index(docs)
    assert idx.count() == 2
    assert idx.contains(1)
    assert idx.contains(2)
