"""
Dataset loader — loads and validates evaluation data from JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import (
    EvaluationDataset,
    EvaluationQuery,
    VALID_CATEGORIES,
    VALID_RELEVANCE_GRADES,
)


class DatasetError(Exception):
    """Raised when the evaluation dataset is invalid."""


def load_dataset(
    documents_path: Path,
    queries_path: Path,
) -> EvaluationDataset:
    """
    Load and validate an evaluation dataset from JSON files.

    Parameters
    ----------
    documents_path : Path
        Path to documents.json.
    queries_path : Path
        Path to search_quality.json.

    Returns
    -------
    EvaluationDataset
        Validated dataset ready for evaluation.

    Raises
    ------
    DatasetError
        If the dataset is malformed or invalid.
    """
    documents = _load_documents(documents_path)
    queries = _load_queries(queries_path)
    return EvaluationDataset(queries=tuple(queries), documents=tuple(documents))


def _load_documents(path: Path) -> list[dict]:
    """Load and validate documents.json."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetError(f"Malformed JSON in {path}: {e}") from e
    except FileNotFoundError:
        raise DatasetError(f"Document file not found: {path}")

    if not isinstance(data, list):
        raise DatasetError(f"Expected a JSON array in {path}")

    seen_ids: set[int] = set()
    for i, doc in enumerate(data):
        if "id" not in doc:
            raise DatasetError(f"Document at index {i} missing 'id'")
        if "title" not in doc:
            raise DatasetError(f"Document {doc.get('id')} missing 'title'")
        if "text" not in doc:
            raise DatasetError(f"Document {doc.get('id')} missing 'text'")

        doc_id = doc["id"]
        if not isinstance(doc_id, int) or doc_id < 0:
            raise DatasetError(f"Document ID must be a non-negative integer, got {doc_id}")
        if doc_id in seen_ids:
            raise DatasetError(f"Duplicate document ID: {doc_id}")
        seen_ids.add(doc_id)

    return data


def _load_queries(path: Path) -> list[EvaluationQuery]:
    """Load and validate search_quality.json."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetError(f"Malformed JSON in {path}: {e}") from e
    except FileNotFoundError:
        raise DatasetError(f"Query file not found: {path}")

    if not isinstance(data, dict) or "queries" not in data:
        raise DatasetError(f"Expected a JSON object with 'queries' key in {path}")

    raw_queries = data["queries"]
    if not isinstance(raw_queries, list):
        raise DatasetError("'queries' must be a JSON array")

    seen_ids: set[str] = set()
    queries: list[EvaluationQuery] = []

    for i, q in enumerate(raw_queries):
        # Validate required fields.
        if "id" not in q:
            raise DatasetError(f"Query at index {i} missing 'id'")
        if "query" not in q:
            raise DatasetError(f"Query '{q.get('id')}' missing 'query'")
        if "category" not in q:
            raise DatasetError(f"Query '{q.get('id')}' missing 'category'")
        if "judgments" not in q:
            raise DatasetError(f"Query '{q.get('id')}' missing 'judgments'")

        query_id = str(q["id"])
        query_text = q["query"]
        category = q["category"]
        raw_judgments = q["judgments"]

        # Validate query ID uniqueness.
        if query_id in seen_ids:
            raise DatasetError(f"Duplicate query ID: {query_id}")
        seen_ids.add(query_id)

        # Validate query text.
        if not query_text or not query_text.strip():
            raise DatasetError(f"Query '{query_id}' has empty query text")

        # Validate category.
        if category not in VALID_CATEGORIES:
            raise DatasetError(
                f"Query '{query_id}' has invalid category '{category}'. "
                f"Expected one of: {sorted(VALID_CATEGORIES)}"
            )

        # Validate judgments.
        if not isinstance(raw_judgments, dict):
            raise DatasetError(f"Query '{query_id}' judgments must be a JSON object")

        judgments: dict[int, int] = {}
        for doc_id_str, grade in raw_judgments.items():
            try:
                doc_id = int(doc_id_str)
            except (ValueError, TypeError):
                raise DatasetError(
                    f"Query '{query_id}': invalid document ID '{doc_id_str}'"
                )
            if doc_id < 0:
                raise DatasetError(
                    f"Query '{query_id}': negative document ID {doc_id}"
                )
            if grade not in VALID_RELEVANCE_GRADES:
                raise DatasetError(
                    f"Invalid relevance grade {grade} for document {doc_id} "
                    f"in query '{query_id}'. Expected value between 0 and 3."
                )
            judgments[doc_id] = grade

        queries.append(EvaluationQuery(
            query_id=query_id,
            query=query_text,
            category=category,
            judgments=judgments,
        ))

    return queries
