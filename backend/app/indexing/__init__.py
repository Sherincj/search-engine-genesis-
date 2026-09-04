"""
Indexing package — inverted index, builder, storage, and document indexer.

Public API
----------
InvertedIndex
    Core in-memory data structure (term → posting list).

Posting
    Single entry in a posting list (doc_id, TF, positions).

IndexBuilder
    Batch-construction helper.

IndexStorage
    JSON-based persistence with atomic writes.

DocumentIndexer
    Coordinates document → tokenise → index pipeline.

IndexingResult
    Metadata returned after indexing a document.
"""

from app.indexing.document_indexer import DocumentIndexer, IndexingResult
from app.indexing.index_builder import IndexBuilder
from app.indexing.index_storage import IndexStorage, IndexStorageError
from app.indexing.inverted_index import InvertedIndex, Posting

__all__ = [
    "DocumentIndexer",
    "IndexBuilder",
    "IndexStorage",
    "IndexStorageError",
    "IndexingResult",
    "InvertedIndex",
    "Posting",
]
