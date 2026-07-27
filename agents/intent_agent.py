"""Intent classification agent for the MATLAB Troubleshooting AI.

Classifies each incoming request into one of four intents using Gemini,
so the LangGraph router can direct it to the correct processing path.
"""


from __future__ import annotations

import logging
from pathlib import Path

import google.generativeai as genai

from config.settings import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_VALID_INTENTS: frozenset[str] = frozenset({"GENERAL", "RAG", "IMAGE", "WEB_SEARCH"})

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "intent_prompt.txt"
_CLASSIFICATION_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def classify_intent(user_query: str, has_image: bool) -> str:
    """Classify the user request into a routing intent.

    If the user uploaded an image, IMAGE intent is returned immediately
    without calling Gemini, as visual analysis always takes priority.

    Args:
        user_query: The text question submitted by the user.
        has_image: True if the user uploaded an image alongside the query.

    Returns:
        One of: ``\"GENERAL\"``, ``\"RAG\"``, ``\"IMAGE\"``, ``\"WEB_SEARCH\"``.
    """
    if has_image:
        logger.debug("Image present — returning IMAGE intent without LLM call.")
        return "IMAGE"

    cfg = get_settings()
    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(cfg.gemini_model)

    prompt = f"{_CLASSIFICATION_PROMPT}\n\nUser Query:\n{user_query}"

    try:
        response = model.generate_content(prompt)
        intent = getattr(response, "text", "").strip().upper()

        if intent not in _VALID_INTENTS:
            logger.warning(
                "Unexpected intent '%s' from model — defaulting to GENERAL.", intent
            )
            return "GENERAL"

        logger.debug("Intent classified: %s", intent)
        return intent

    except Exception as exc:  # noqa: BLE001
        logger.error("Intent classification failed: %s — defaulting to GENERAL.", exc)
        return "GENERAL"
