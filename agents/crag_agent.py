"""Corrective RAG (CRAG) agent for the MATLAB Troubleshooting AI.

When retrieval grading returns BAD, this agent rewrites the query to
improve the next retrieval attempt. Retry counting is delegated to the
graph node so this function remains a pure transformation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import google.generativeai as genai

from config.settings import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "crag_prompt.txt"
_REWRITE_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def rewrite_query(query: str) -> str:
    """Rewrite a query to improve retrieval performance.

    Args:
        query: The current query — either the original user query or
            a previously rewritten version.

    Returns:
        An improved query string. Falls back to the original query if
        the Gemini call fails, ensuring the graph never stalls.
    """
    cfg = get_settings()
    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(cfg.gemini_model)

    prompt = _REWRITE_PROMPT.format(query=query)

    try:
        response = model.generate_content(prompt)
        rewritten = response.text.strip()

        if not rewritten:
            logger.warning("Empty rewrite returned — keeping original query.")
            return query

        logger.info("Query rewritten: '%s' → '%s'", query, rewritten)
        return rewritten

    except Exception as exc:  # noqa: BLE001
        logger.error("Query rewrite failed: %s — returning original query.", exc)
        return query
