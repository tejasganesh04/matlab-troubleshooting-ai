"""PDF document loader using PyPDFLoader."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger: logging.Logger = logging.getLogger(__name__)


def load_pdfs_from_directory(directory: Path) -> list[Document]:
    """Load all PDF files from the specified directory.

    Args:
        directory: The directory containing PDF files.

    Returns:
        A list of LangChain Document objects.
    """
    documents: list[Document] = []
    if not directory.exists() or not directory.is_dir():
        logger.warning("PDF directory does not exist: %s", directory)
        return documents

    for pdf_file in directory.glob("*.pdf"):
        logger.info("Loading PDF: %s", pdf_file.name)
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s", pdf_file.name, exc)
            
    logger.info("Loaded %d pages from PDFs.", len(documents))
    return documents
