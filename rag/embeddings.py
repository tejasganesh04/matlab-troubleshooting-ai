"""Embeddings module for the RAG pipeline."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import get_settings


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Initialize and return the Google Generative AI embeddings model.

    Returns:
        An instance of GoogleGenerativeAIEmbeddings configured with the
        application settings.
    """
    cfg = get_settings()
    
    return GoogleGenerativeAIEmbeddings(
        model=cfg.embedding_model,
        google_api_key=cfg.gemini_api_key,
    )
