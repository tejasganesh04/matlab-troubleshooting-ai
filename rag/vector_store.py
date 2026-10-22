"""FAISS vector store creation, saving, loading, and searching."""

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger: logging.Logger = logging.getLogger(__name__)


def create_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    save_path: Path,
) -> FAISS:
    """Create a FAISS vector store from documents and save it.

    Args:
        documents: List of Document chunks.
        embeddings: The embedding model to use.
        save_path: Directory path where the FAISS index should be saved.

    Returns:
        The created FAISS vector store instance.
    """
    logger.info("Creating FAISS vector store from %d documents.", len(documents))
    vector_store = FAISS.from_documents(documents, embeddings)
    
    logger.info("Saving FAISS index to %s", save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))
    
    return vector_store


def load_vector_store(
    load_path: Path,
    embeddings: Embeddings,
) -> FAISS:
    """Load a FAISS vector store from disk.

    Args:
        load_path: Directory path where the FAISS index is saved.
        embeddings: The embedding model used for the index.

    Returns:
        The loaded FAISS vector store instance.
    """
    logger.info("Loading FAISS index from %s", load_path)
    return FAISS.load_local(str(load_path), embeddings, allow_dangerous_deserialization=True)
