"""Retriever wrapper for querying the FAISS vector store."""

from __future__ import annotations

import logging

from config.settings import get_settings
from rag.embeddings import get_embeddings_model
from rag.vector_store import load_vector_store

logger: logging.Logger = logging.getLogger(__name__)


def retrieve_documents(
    query: str,
    top_k: int = 5,
) -> tuple[list[str], str]:
    """Retrieve relevant documents from the FAISS index.

    Args:
        query: The user's search query.
        top_k: Maximum number of document chunks to retrieve.

    Returns:
        A tuple containing:
            - list[str]: Retrieved document texts.
            - str: Concatenated retrieval context.
    """
    cfg = get_settings()

    logger.info("Loading FAISS vector store.")
    embeddings = get_embeddings_model()
    vector_store = load_vector_store(
        cfg.faiss_index_dir,
        embeddings,
    )

    logger.info("Retrieving top %d documents.", top_k)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k}
    )

    docs = retriever.invoke(query)

    documents = [
        doc.page_content
        for doc in docs
    ]

    context = "\n\n".join(documents)

    logger.info("Retrieved %d document chunks.", len(documents))

    return documents, context