"""
Indexing package — inverted index, field index, builder, storage, and document indexer.

Public API
----------
InvertedIndex
    Core in-memory data structure (term → posting list).

FieldIndex
    Field-aware wrapper (title + body InvertedIndex instances).

Posting
    Single entry in a posting list (doc_id, TF, positions).

IndexBuilder
    Batch-construction helper (single-field, backward compatible).

FieldIndexBuilder
    Batch-construction helper (field-aware).

IndexStorage
    JSON-based persistence with atomic writes (field-aware, version 2).

DocumentIndexer
    Coordinates document → tokenise → field-aware index pipeline.

IndexingResult
    Metadata returned after indexing a document.
"""

from app.indexing.document_indexer import DocumentIndexer, IndexingResult
from app.indexing.field_index import FIELD_BODY, FIELD_TITLE, FieldIndex
from app.indexing.index_builder import FieldIndexBuilder, IndexBuilder
from app.indexing.index_storage import IndexStorage, IndexStorageError
from app.indexing.inverted_index import InvertedIndex, Posting

__all__ = [
    "DocumentIndexer",
    "FIELD_BODY",
    "FIELD_TITLE",
    "FieldIndex",
    "FieldIndexBuilder",
    "IndexBuilder",
    "IndexStorage",
    "IndexStorageError",
    "IndexingResult",
    "InvertedIndex",
    "Posting",
]
