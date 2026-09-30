"""Tests for vector index persistence."""

import json
import pytest
import numpy as np

from app.vector_index.vector_index import VectorIndex
from app.vector_index.vector_storage import VectorStorage, VectorStorageError, VECTOR_INDEX_VERSION


def test_save_and_load(tmp_path):
    idx = VectorIndex(dimension=3)
    idx.add(1, [0.1, 0.2, 0.3])
    idx.add(2, [0.4, 0.5, 0.6])

    storage = VectorStorage(tmp_path / "vec_idx")
    storage.save(idx, model_name="test-model", normalize=True)

    loaded, meta = storage.load()
    assert loaded.count() == 2
    assert loaded.contains(1)
    assert loaded.contains(2)
    assert loaded.dimension == 3
    assert meta.model_name == "test-model"
    assert meta.dimension == 3
    assert meta.normalize is True
    assert meta.version == VECTOR_INDEX_VERSION


def test_save_empty_index(tmp_path):
    idx = VectorIndex(dimension=4)
    storage = VectorStorage(tmp_path / "empty")
    storage.save(idx, model_name="m", normalize=False)
    loaded, meta = storage.load()
    assert loaded.count() == 0
    assert loaded.dimension == 4


def test_exists(tmp_path):
    storage = VectorStorage(tmp_path / "vec")
    assert storage.exists() is False
    idx = VectorIndex(dimension=2)
    storage.save(idx, model_name="m", normalize=True)
    assert storage.exists() is True


def test_roundtrip_vectors(tmp_path):
    idx = VectorIndex(dimension=3)
    idx.add(10, [1.0, 2.0, 3.0])
    storage = VectorStorage(tmp_path / "rt")
    storage.save(idx, model_name="m", normalize=True)
    loaded, _ = storage.load()
    vec = loaded.get_vector(10)
    assert vec is not None
    assert len(vec) == 3
    assert abs(vec[0] - 1.0) < 1e-5
    assert abs(vec[1] - 2.0) < 1e-5
    assert abs(vec[2] - 3.0) < 1e-5


def test_incompatible_model(tmp_path):
    idx = VectorIndex(dimension=3)
    idx.add(1, [0.1, 0.2, 0.3])
    storage = VectorStorage(tmp_path / "model")
    storage.save(idx, model_name="model-A", normalize=True)
    with pytest.raises(VectorStorageError, match="Incompatible embedding model"):
        storage.load(expected_model="model-B")


def test_incompatible_dimension(tmp_path):
    idx = VectorIndex(dimension=3)
    idx.add(1, [0.1, 0.2, 0.3])
    storage = VectorStorage(tmp_path / "dim")
    storage.save(idx, model_name="m", normalize=True)
    with pytest.raises(VectorStorageError, match="Incompatible vector dimension"):
        storage.load(expected_dimension=768)


def test_incompatible_normalization(tmp_path):
    idx = VectorIndex(dimension=2)
    storage = VectorStorage(tmp_path / "normalization")
    storage.save(idx, model_name="m", normalize=True)
    with pytest.raises(VectorStorageError, match="normalization"):
        storage.load(expected_normalize=False)


def test_incompatible_version(tmp_path):
    idx = VectorIndex(dimension=2)
    storage = VectorStorage(tmp_path / "ver")
    storage.save(idx, model_name="m", normalize=True)
    # Tamper with version
    meta_path = tmp_path / "ver" / "metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta["version"] = 999
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(VectorStorageError, match="Incompatible vector index version"):
        storage.load()


def test_missing_metadata(tmp_path):
    storage = VectorStorage(tmp_path / "missing")
    with pytest.raises(VectorStorageError, match="Metadata file not found"):
        storage.load()


def test_corrupt_metadata(tmp_path):
    path = tmp_path / "corrupt"
    path.mkdir()
    (path / "metadata.json").write_text("{{invalid json")
    (path / "vectors.npz").write_text("")
    with pytest.raises(VectorStorageError, match="Corrupt metadata"):
        storage = VectorStorage(path)
        storage.load()


def test_corrupt_vectors(tmp_path):
    path = tmp_path / "corrvec"
    path.mkdir()
    meta = {"version": VECTOR_INDEX_VERSION, "model_name": "m", "dimension": 3, "normalize": True, "document_count": 0}
    with open(path / "metadata.json", "w") as f:
        json.dump(meta, f)
    (path / "vectors.npz").write_text("not a real npz")
    with pytest.raises(VectorStorageError, match="Corrupt vectors"):
        storage = VectorStorage(path)
        storage.load()


def test_metadata_count_must_match_vectors(tmp_path):
    idx = VectorIndex(dimension=2)
    idx.add(1, [1.0, 0.0])
    storage = VectorStorage(tmp_path / "count")
    storage.save(idx, model_name="m", normalize=True)
    meta_path = tmp_path / "count" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["document_count"] = 2
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(VectorStorageError, match="document count"):
        storage.load()
