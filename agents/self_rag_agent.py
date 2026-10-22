"""Self-RAG (Self-Reflective Retrieval-Augmented Generation) agent.

Verifies whether the generated answer is grounded in the available context
and accurately addresses the user's question. Returns PASS or FAIL to
drive the Self-RAG routing decision in the graph.
"""

from __future__ import annotations

import logging
from pathlib import Path

import google.generativeai as genai

from config.settings import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_VALID_VERDICTS: frozenset[str] = frozenset({"PASS", "FAIL"})

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "self_rag_prompt.txt"
_SELF_RAG_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def verify_answer(query: str, context: str, answer: str) -> str:
    """Verify whether the generated answer is grounded and accurate.

    An empty answer immediately returns FAIL. A Gemini API failure
    defaults to PASS to avoid unnecessary retries on infrastructure errors.

    Args:
        query: The user's original question.
        context: The combined context that was used during generation.
        answer: The generated answer to evaluate.

    Returns:
        ``\"PASS\"`` if the answer is grounded, ``\"FAIL\"`` otherwise.
    """
    if not answer.strip():
        logger.warning("Empty answer submitted to Self-RAG — returning FAIL.")
        return "FAIL"

    cfg = get_settings()
    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(cfg.gemini_model)

    prompt = _SELF_RAG_PROMPT.format(
        query=query,
        context=context.strip() if context.strip() else "No context available.",
        answer=answer,
    )

    try:
        response = model.generate_content(prompt)
        verdict = response.text.strip().upper()

        if verdict not in _VALID_VERDICTS:
            logger.warning(
                "Unexpected verdict '%s' from model — defaulting to PASS.", verdict
            )
            return "PASS"

        logger.info("Self-RAG verdict: %s", verdict)
        return verdict

    except Exception as exc:  # noqa: BLE001
        logger.error("Self-RAG verification failed: %s — defaulting to PASS.", exc)
        return "PASS"
