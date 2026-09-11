"""
Tests for IndexStorage — save, load, and round-trip.

Uses a temporary directory for each test to avoid polluting real data.
"""

import json
from pathlib import Path

import pytest

from app.indexing.index_builder import FieldIndexBuilder
from app.indexing.index_storage import IndexStorage, IndexStorageError
from app.indexing.field_index import FieldIndex


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary directory for index storage."""
    return tmp_path / "test_index"


@pytest.fixture
def storage(tmp_index_dir: Path) -> IndexStorage:
    """IndexStorage pointing at a temporary directory."""
    return IndexStorage(tmp_index_dir)


# ── helpers ──────────────────────────────────────────────────

def _build_sample_index() -> FieldIndex:
    return FieldIndexBuilder.from_documents([
        (1, [], ["python", "web", "python"]),
        (2, [], ["javascript", "web"]),
        (3, [], ["python", "api", "framework"]),
    ])


# ── exists / not exists ─────────────────────────────────────

def test_exists_false_initially(storage: IndexStorage):
    assert storage.exists() is False


def test_exists_true_after_save(storage: IndexStorage):
    storage.save(FieldIndex())
    assert storage.exists() is True


# ── save + load empty index ──────────────────────────────────

def test_save_empty_index(storage: IndexStorage):
    idx = FieldIndex()
    storage.save(idx)
    assert storage.exists()


def test_load_empty_index(storage: IndexStorage):
    storage.save(FieldIndex())
    loaded = storage.load()

    assert loaded.get_total_documents() == 0
    assert loaded.vocabulary_size == 0
    assert loaded.get_average_document_length("body") == 0.0


# ── round-trip with data ────────────────────────────────────

def test_round_trip_total_documents(storage: IndexStorage):
    original = _build_sample_index()
    storage.save(original)
    loaded = storage.load()

    assert loaded.get_total_documents() == original.get_total_documents()


def test_round_trip_vocabulary(storage: IndexStorage):
    original = _build_sample_index()
    storage.save(original)
    loaded = storage.load()

    assert loaded.vocabulary == original.vocabulary


def test_round_trip_postings(storage: IndexStorage):
    original = _build_sample_index()
    storage.save(original)
    loaded = storage.load()

    for term in original.vocabulary:
        orig_postings = original.body_index.get_postings(term)
        load_postings = loaded.body_index.get_postings(term)
        assert len(load_postings) == len(orig_postings)

        for op, lp in zip(orig_postings, load_postings):
            assert lp.document_id == op.document_id
            assert lp.term_frequency == op.term_frequency
            assert lp.positions == op.positions


def test_round_trip_document_lengths(storage: IndexStorage):
    original = _build_sample_index()
    storage.save(original)
    loaded = storage.load()

    for doc_id in [1, 2, 3]:
        assert loaded.get_document_length(doc_id, "body") == \
               original.get_document_length(doc_id, "body")


def test_round_trip_document_frequencies(storage: IndexStorage):
    original = _build_sample_index()
    storage.save(original)
    loaded = storage.load()

    for term in original.vocabulary:
        assert loaded.get_document_frequency(term, "body") == \
               original.get_document_frequency(term, "body")


def test_round_trip_average_length(storage: IndexStorage):
    original = _build_sample_index()
    storage.save(original)
    loaded = storage.load()

    assert loaded.get_average_document_length("body") == \
           original.get_average_document_length("body")


def test_round_trip_validation(storage: IndexStorage):
    """The loaded index must be internally consistent."""
    original = _build_sample_index()
    storage.save(original)
    loaded = storage.load()

    assert loaded.validate() == []


# ── overwrite existing index ─────────────────────────────────

def test_overwrite(storage: IndexStorage):
    idx1 = FieldIndexBuilder.from_documents([(1, [], ["python"])])
    storage.save(idx1)

    idx2 = FieldIndexBuilder.from_documents([
        (10, [], ["javascript", "web"]),
        (20, [], ["rust"]),
    ])
    storage.save(idx2)

    loaded = storage.load()
    assert loaded.get_total_documents() == 2
    assert loaded.body_index.contains_term("javascript")
    assert not loaded.body_index.contains_term("python")


# ── load missing index ──────────────────────────────────────

def test_load_missing_raises(storage: IndexStorage):
    with pytest.raises(IndexStorageError, match="not found"):
        storage.load()


# ── corrupt index ────────────────────────────────────────────

def test_load_corrupt_raises(storage: IndexStorage, tmp_index_dir: Path):
    # Create a corrupt file
    tmp_index_dir.mkdir(parents=True, exist_ok=True)
    (tmp_index_dir / "index.json").write_text("{invalid json!!")

    with pytest.raises(IndexStorageError, match="Corrupt"):
        storage.load()


def test_load_old_format_raises(storage: IndexStorage, tmp_index_dir: Path):
    """An old Phase 7 index format lacking 'index_version' should be rejected."""
    tmp_index_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp_index_dir / "index.json", "w") as f:
        # Pre-field-aware format missing index_version and fields
        json.dump({"_postings": {}, "_doc_lengths": {}, "_doc_terms": {}}, f)

    with pytest.raises(IndexStorageError, match="format is outdated"):
        storage.load()


# ── delete ───────────────────────────────────────────────────

def test_delete(storage: IndexStorage):
    storage.save(FieldIndex())
    assert storage.exists()

    storage.delete()
    assert not storage.exists()


def test_delete_nonexistent_is_safe(storage: IndexStorage):
    """Deleting when nothing exists should not raise."""
    storage.delete()
