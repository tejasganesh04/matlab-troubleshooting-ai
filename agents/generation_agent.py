"""Generation agent for the MATLAB Troubleshooting AI.

Assembles all available context (vision, retrieval, web) and uses Gemini
to produce a grounded, technically precise MATLAB troubleshooting response.
"""

from __future__ import annotations

import logging
from pathlib import Path

import google.generativeai as genai

from config.settings import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"
_GENERATION_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

_CONTEXT_BLOCK = """Available Context:
{context}

Ground your response in the above context."""

_NO_CONTEXT_BLOCK = "No external context is available. Use your general MATLAB and Simulink expertise."


def _assemble_context(
    retrieved_context: str,
    vision_context: str,
    web_context: str,
) -> str:
    """Combine all available context sources into a single string.

    Args:
        retrieved_context: Context from FAISS document retrieval.
        vision_context: Structured text from the Vision Agent.
        web_context: Text results from the Web Search Agent.

    Returns:
        A labelled, concatenated context string, or empty string if none exist.
    """
    sections: list[str] = []
    if vision_context.strip():
        sections.append(f"[Image Analysis]\n{vision_context.strip()}")
    if retrieved_context.strip():
        sections.append(f"[Documentation]\n{retrieved_context.strip()}")
    if web_context.strip():
        sections.append(f"[Web Search]\n{web_context.strip()}")
    return "\n\n".join(sections)


def generate_answer(
    query: str,
    retrieved_context: str = "",
    vision_context: str = "",
    web_context: str = "",
) -> str:
    """Generate a MATLAB troubleshooting answer using Gemini.

    Args:
        query: The user's question.
        retrieved_context: Context from FAISS retrieval.
        vision_context: Context from the Vision Agent.
        web_context: Context from the Web Search Agent.

    Returns:
        The generated answer as a plain text string.

    Raises:
        RuntimeError: If the Gemini API call fails.
    """
    cfg = get_settings()
    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(cfg.gemini_model)

    combined = _assemble_context(retrieved_context, vision_context, web_context)
    context_block = (
        _CONTEXT_BLOCK.format(context=combined) if combined else _NO_CONTEXT_BLOCK
    )

    prompt = _GENERATION_PROMPT.format(context_block=context_block, query=query)

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logger.info("Answer generated — %d characters.", len(answer))
        return answer

    except Exception as exc:  # noqa: BLE001
        logger.error("Generation failed: %s", exc)
        raise RuntimeError(f"Generation failed: {exc}") from exc
