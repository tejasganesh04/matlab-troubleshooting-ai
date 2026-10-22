"""Ingestion and Indexing pipeline executor."""

from __future__ import annotations

import logging

from config.settings import get_settings
from rag.embeddings import get_embeddings_model
from rag.pdf_loader import load_pdfs_from_directory
from rag.text_splitter import split_documents
from rag.vector_store import create_vector_store

logger: logging.Logger = logging.getLogger(__name__)


def run_indexing_pipeline() -> None:
    """Run the full ingestion pipeline."""

    cfg = get_settings()

    logger.info("Starting RAG indexing pipeline.")

    # 1. Load PDFs
    documents = load_pdfs_from_directory(cfg.knowledge_base_dir)

    if not documents:
        logger.warning("No documents loaded. Aborting indexing pipeline.")
        return

    # 2. Split documents
    chunks = split_documents(documents)

    # 3. Load embedding model
    embeddings_model = get_embeddings_model()

    # 4. Create FAISS index
    create_vector_store(
        chunks,
        embeddings_model,
        cfg.faiss_index_dir,
    )

    logger.info("RAG indexing pipeline completed successfully.")


if __name__ == "__main__":
    run_indexing_pipeline()