"""
Build vector index — CLI tool to generate and persist embeddings for all documents.

Loads all documents from the SQLite database, generates embeddings in batches,
builds the VectorIndex, and saves it to disk.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure the backend directory is on sys.path.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config.settings import settings
from app.database.database import SessionLocal, init_db
from app.database.repositories import DocumentRepository
from app.embeddings.sentence_transformer import SentenceTransformerProvider
from app.indexing.semantic_indexer import SemanticIndexer
from app.vector_index.vector_index import VectorIndex
from app.vector_index.vector_storage import VectorStorage
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def build_index() -> None:
    print("Initializing database...")
    await init_db()

    print("Loading documents...")
    async with SessionLocal() as db:
        repo = DocumentRepository(db)
        documents = await repo.get_all_documents()

    if not documents:
        print("No documents found in database.")
        return

    doc_count = len(documents)
    print(f"Documents: {doc_count}\n")

    print(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
    provider = SentenceTransformerProvider(
        model_name=settings.EMBEDDING_MODEL,
        normalize=settings.EMBEDDING_NORMALIZE,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    dim = provider.dimension
    print(f"Dimension: {dim}\n")

    print("Embedding documents...")
    index = VectorIndex(dimension=dim)
    indexer = SemanticIndexer(index, provider)

    # Convert to tuples for bulk indexing
    doc_tuples = [(d.id, d.title, d.text) for d in documents]

    # Process in batches to show progress
    batch_size = settings.EMBEDDING_BATCH_SIZE
    processed = 0

    for i in range(0, doc_count, batch_size):
        batch = doc_tuples[i:i + batch_size]
        indexer.bulk_index(batch)
        processed += len(batch)
        
        # Simple progress bar
        progress = int((processed / doc_count) * 20)
        bar = "█" * progress + " " * (20 - progress)
        print(f"\r[{bar}] {processed}/{doc_count}", end="", flush=True)

    print("\n\nSaving index...")
    storage = VectorStorage(
        settings.VECTOR_INDEX_PATH,
        version=settings.VECTOR_INDEX_VERSION,
    )
    storage.save(
        index,
        model_name=settings.EMBEDDING_MODEL,
        normalize=settings.EMBEDDING_NORMALIZE,
    )

    print("\nVector index built successfully.")
    print(f"Documents indexed: {index.count()}")
    print(f"Dimension: {dim}")
    print(f"Model: {settings.EMBEDDING_MODEL}")


def main() -> None:
    setup_logging(debug=False)
    
    # Silence noisy loggers
    import logging
    for noisy in ["httpcore", "httpx", "aiosqlite", "sqlalchemy"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        asyncio.run(build_index())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
