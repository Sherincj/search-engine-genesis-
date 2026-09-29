import json
from pathlib import Path
import pytest

from evaluation.dataset import load_dataset, DatasetError

EVAL_DATA_DIR = Path(__file__).parent.parent / "evaluation_data"
DOCS_PATH = EVAL_DATA_DIR / "documents.json"
QUERIES_PATH = EVAL_DATA_DIR / "search_quality.json"

def test_load_valid_dataset():
    dataset = load_dataset(DOCS_PATH, QUERIES_PATH)
    assert len(dataset.documents) > 0
    assert len(dataset.queries) > 0

def test_malformed_json(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json")
    
    with pytest.raises(DatasetError, match="Malformed JSON"):
        load_dataset(bad_json, QUERIES_PATH)
        
    with pytest.raises(DatasetError, match="Malformed JSON"):
        load_dataset(DOCS_PATH, bad_json)

def test_duplicate_query_ids(tmp_path):
    queries = {
        "queries": [
            {"id": "q1", "query": "a", "category": "single_term", "judgments": {}},
            {"id": "q1", "query": "b", "category": "single_term", "judgments": {}}
        ]
    }
    q_file = tmp_path / "q.json"
    q_file.write_text(json.dumps(queries))
    with pytest.raises(DatasetError, match="Duplicate query ID"):
        load_dataset(DOCS_PATH, q_file)

def test_invalid_relevance_grade(tmp_path):
    queries = {
        "queries": [
            {"id": "q1", "query": "a", "category": "single_term", "judgments": {"1": 7}}
        ]
    }
    q_file = tmp_path / "q.json"
    q_file.write_text(json.dumps(queries))
    with pytest.raises(DatasetError, match="Invalid relevance grade"):
        load_dataset(DOCS_PATH, q_file)

def test_missing_query_field(tmp_path):
    queries = {
        "queries": [
            {"id": "q1", "category": "single_term", "judgments": {}}
        ]
    }
    q_file = tmp_path / "q.json"
    q_file.write_text(json.dumps(queries))
    with pytest.raises(DatasetError, match="missing 'query'"):
        load_dataset(DOCS_PATH, q_file)

def test_empty_query_text(tmp_path):
    queries = {
        "queries": [
            {"id": "q1", "query": "  ", "category": "single_term", "judgments": {}}
        ]
    }
    q_file = tmp_path / "q.json"
    q_file.write_text(json.dumps(queries))
    with pytest.raises(DatasetError, match="empty query text"):
        load_dataset(DOCS_PATH, q_file)

def test_invalid_category(tmp_path):
    queries = {
        "queries": [
            {"id": "q1", "query": "a", "category": "invalid_cat", "judgments": {}}
        ]
    }
    q_file = tmp_path / "q.json"
    q_file.write_text(json.dumps(queries))
    with pytest.raises(DatasetError, match="invalid category"):
        load_dataset(DOCS_PATH, q_file)

def test_negative_doc_id(tmp_path):
    docs = [
        {"id": -1, "title": "t", "text": "t"}
    ]
    d_file = tmp_path / "d.json"
    d_file.write_text(json.dumps(docs))
    with pytest.raises(DatasetError, match="non-negative integer"):
        load_dataset(d_file, QUERIES_PATH)
        
    queries = {
        "queries": [
            {"id": "q1", "query": "a", "category": "single_term", "judgments": {"-1": 1}}
        ]
    }
    q_file = tmp_path / "q.json"
    q_file.write_text(json.dumps(queries))
    with pytest.raises(DatasetError, match="negative document ID"):
        load_dataset(DOCS_PATH, q_file)

def test_duplicate_document_ids(tmp_path):
    docs = [
        {"id": 1, "title": "t", "text": "t"},
        {"id": 1, "title": "t2", "text": "t2"}
    ]
    d_file = tmp_path / "d.json"
    d_file.write_text(json.dumps(docs))
    with pytest.raises(DatasetError, match="Duplicate document ID"):
        load_dataset(d_file, QUERIES_PATH)

def test_missing_document_fields(tmp_path):
    docs = [
        {"id": 1, "text": "t"}
    ]
    d_file = tmp_path / "d.json"
    d_file.write_text(json.dumps(docs))
    with pytest.raises(DatasetError, match="missing 'title'"):
        load_dataset(d_file, QUERIES_PATH)
