"""RAG package for the MATLAB Troubleshooting AI."""

from rag.embeddings import get_embeddings_model
from rag.indexer import run_indexing_pipeline
from rag.pdf_loader import load_pdfs_from_directory
from rag.retriever import retrieve_documents
from rag.text_splitter import split_documents
from rag.vector_store import (
    create_vector_store,
    load_vector_store,
)

__all__ = [
    "get_embeddings_model",
    "run_indexing_pipeline",
    "load_pdfs_from_directory",
    "retrieve_documents",
    "split_documents",
    "create_vector_store",
    "load_vector_store",
]