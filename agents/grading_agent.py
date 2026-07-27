"""Retrieval grading agent for the MATLAB Troubleshooting AI.

Evaluates whether the retrieved context is sufficient to answer the
user's question. Returns GOOD or BAD to drive the CRAG routing decision.
"""

from __future__ import annotations

import logging
from pathlib import Path

import google.generativeai as genai

from config.settings import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_VALID_GRADES: frozenset[str] = frozenset({"GOOD", "BAD"})

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "grading_prompt.txt"
_GRADING_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def grade_retrieval(query: str, context: str) -> str:
    """Grade the quality of retrieved context against the user query.

    If context is empty, BAD is returned immediately without calling Gemini.

    Args:
        query: The user's original or rewritten question.
        context: The concatenated retrieved document context.

    Returns:
        ``\"GOOD\"`` if context is sufficient, ``\"BAD\"`` otherwise.
    """
    if not context.strip():
        logger.info("Empty context — grading as BAD without LLM call.")
        return "BAD"

    cfg = get_settings()
    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(cfg.gemini_model)

    prompt = _GRADING_PROMPT.format(query=query, context=context)

    try:
        response = model.generate_content(prompt)
        grade = response.text.strip().upper()

        if grade not in _VALID_GRADES:
            logger.warning(
                "Unexpected grade '%s' from model — defaulting to BAD.", grade
            )
            return "BAD"

        logger.info("Retrieval grade: %s", grade)
        return grade

    except Exception as exc:  # noqa: BLE001
        logger.error("Grading failed: %s — defaulting to BAD.", exc)
        return "BAD"
