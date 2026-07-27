"""Retrieval agent for the MATLAB Troubleshooting AI.

Loads the FAISS vector store and retrieves the most relevant document
chunks for a given query. Falls back gracefully when the index does
not exist yet (i.e. before the indexing pipeline has been run).
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import get_settings
from rag.embeddings import get_embeddings_model
from rag.retriever import retrieve_documents as faiss_retrieve
from rag.vector_store import load_vector_store

logger: logging.Logger = logging.getLogger(__name__)

# Module-level cache so the vector store is only loaded once per process
_vector_store = None


def _get_vector_store():
    """Load and cache the FAISS vector store.

    Returns:
        The loaded FAISS vector store, or None if the index does not exist.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    cfg = get_settings()
    index_path: Path = cfg.faiss_index_dir

    # Check that the FAISS index files exist before trying to load
    if not index_path.exists() or not any(index_path.iterdir()):
        logger.warning(
            "FAISS index not found at '%s'. Run the indexing pipeline first. "
            "Returning empty retrieval.",
            index_path,
        )
        return None

    try:
        embeddings = get_embeddings_model()
        _vector_store = load_vector_store(index_path, embeddings)
        logger.info("FAISS vector store loaded from '%s'.", index_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load FAISS vector store: %s — returning empty retrieval.", exc)
        return None

    return _vector_store


def retrieve_documents(query: str, top_k: int = 5) -> tuple[list[str], str]:
    """Retrieve relevant documents from the MATLAB knowledge base.

    Args:
        query: The search query — either the original user query or a
            CRAG-rewritten query.
        top_k: Maximum number of document chunks to retrieve.

    Returns:
        A tuple of:
            - ``list[str]``: Individual document chunk strings.
            - ``str``: All chunks joined into a single context string.
    """
    vector_store = _get_vector_store()

    if vector_store is None:
        logger.warning("No vector store available — returning empty retrieval.")
        return [], ""

    try:
        docs = faiss_retrieve(query=query, vector_store=vector_store, top_k=top_k)
        doc_strings = [doc.page_content for doc in docs]
        context = "\n\n".join(doc_strings)
        logger.info("Retrieved %d document chunks for query.", len(doc_strings))
        return doc_strings, context
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed: %s — returning empty.", exc)
        return [], ""
