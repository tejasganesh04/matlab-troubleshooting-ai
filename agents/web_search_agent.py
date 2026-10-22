"""Web Search agent for the MATLAB Troubleshooting AI.

Searches the web for recent MATLAB/Simulink information using Tavily
and returns a concise text context for the Generation Agent.

The agent first uses Gemini (via web_search_prompt.txt) to rewrite the
user query into an optimised Tavily search string before issuing the
API call, mirroring the CRAG pattern used for FAISS retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import google.generativeai as genai
from tavily import TavilyClient

from config.settings import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "web_search_prompt.txt"
_WEB_SEARCH_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def _optimise_query(query: str) -> str:
    """Rewrite the user query into an optimised Tavily search string.

    Uses the web_search_prompt.txt template and Gemini to produce a
    concise, keyword-rich query. Falls back to the original query if
    the LLM call fails so the agent never blocks the graph.

    Args:
        query: Raw user query or CRAG-rewritten query.

    Returns:
        Optimised search query string.
    """
    cfg = get_settings()
    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(cfg.gemini_model)

    prompt = _WEB_SEARCH_PROMPT.format(query=query)

    try:
        response = model.generate_content(prompt)
        optimised = response.text.strip()

        if not optimised:
            logger.warning("Empty optimised query returned — using original.")
            return query

        logger.info("Web search query optimised: '%s' → '%s'", query, optimised)
        return optimised

    except Exception as exc:  # noqa: BLE001
        logger.warning("Query optimisation failed: %s — using original query.", exc)
        return query


def search_web(query: str) -> str:
    """Search the web for MATLAB-related information.

    The user query is first rewritten by Gemini into an optimised search
    string (via web_search_prompt.txt) before being sent to Tavily.

    Args:
        query: User query or rewritten query.

    Returns:
        Concise web search context string, or empty string on no results.

    Raises:
        RuntimeError: If the Tavily API request fails.
    """
    cfg = get_settings()
    client = TavilyClient(api_key=cfg.tavily_api_key)

    optimised_query = _optimise_query(query)

    try:
        response = client.search(
            query=optimised_query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False,
        )

        context_parts: list[str] = []

        answer = response.get("answer", "").strip()
        if answer:
            context_parts.append(answer)

        for result in response.get("results", []):
            title = result.get("title", "").strip()
            content = result.get("content", "").strip()

            if title or content:
                context_parts.append(f"{title}\n{content}" if title else content)

        web_context = "\n\n".join(context_parts).strip()

        if not web_context:
            logger.warning("Web Search returned no usable results.")
            return ""

        logger.info(
            "Web Search complete — %d characters retrieved.",
            len(web_context),
        )

        return web_context

    except Exception as exc:  # noqa: BLE001
        logger.error("Web Search failed: %s", exc)
        raise RuntimeError(f"Web Search failed: {exc}") from exc