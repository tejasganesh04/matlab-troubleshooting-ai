"""Text splitting using RecursiveCharacterTextSplitter."""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger: logging.Logger = logging.getLogger(__name__)


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of loaded Document objects.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A list of chunked Document objects.
    """
    logger.info("Splitting %d documents.", len(documents))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info("Created %d chunks.", len(chunks))
    return chunks
