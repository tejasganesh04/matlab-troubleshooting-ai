"""LangGraph conditional edge routing functions.

Each function here is a router that reads the current GraphState and
returns a string key that LangGraph uses to select the next node.
All retry limit enforcement is centralised here so it cannot be
accidentally bypassed by individual agent nodes.
"""

from __future__ import annotations

import logging

from graph.state import GraphState, MAX_RETRIES

logger: logging.Logger = logging.getLogger(__name__)


def route_by_intent(state: GraphState) -> str:
    """Select the next node based on the classified intent.

    Args:
        state: Current graph state.

    Returns:
        One of: ``\"vision_agent\"``, ``\"retrieval_agent\"``,
        ``\"web_search_agent\"``, ``\"generation_agent\"``.
    """
    intent = state.get("intent", "GENERAL")
    logger.debug("Routing by intent: %s", intent)

    routes: dict[str, str] = {
        "GENERAL":    "generation_agent",
        "IMAGE":      "vision_agent",
        "WEB_SEARCH": "web_search_agent",
        "RAG":        "retrieval_agent",
    }
    destination = routes.get(intent, "generation_agent")
    logger.info("Intent '%s' → %s", intent, destination)
    return destination


def route_by_grading(state: GraphState) -> str:
    """Select the next node based on retrieval grading result.

    Args:
        state: Current graph state.

    Returns:
        ``\"web_search_agent\"`` if grade is GOOD (proceed to enrich + generate),
        ``\"crag_agent\"`` if BAD.
    """
    grade = state.get("retrieval_grade", "BAD")
    logger.debug("Routing by retrieval grade: %s", grade)

    if grade == "GOOD":
        logger.info("Retrieval grade GOOD → web_search_agent")
        return "web_search_agent"

    logger.info("Retrieval grade BAD → crag_agent")
    return "crag_agent"


def route_after_crag(state: GraphState) -> str:
    """Select the next node after CRAG query rewriting.

    Allows another retrieval attempt while crag_retry_count is below the
    maximum. Once exhausted, routes directly to web_search_agent so the
    graph always terminates and still attempts generation.

    Args:
        state: Current graph state.

    Returns:
        ``\"retrieval_agent\"`` if retries remain, ``\"web_search_agent\"``
        when the limit is reached.
    """
    crag_retry_count = state.get("crag_retry_count", 0)
    logger.debug("CRAG routing — crag_retry_count=%d / MAX=%d", crag_retry_count, MAX_RETRIES)

    if crag_retry_count < MAX_RETRIES:
        logger.info("CRAG retry %d/%d → retrieval_agent", crag_retry_count, MAX_RETRIES)
        return "retrieval_agent"

    logger.warning(
        "CRAG retry limit reached (%d/%d) → web_search_agent", crag_retry_count, MAX_RETRIES
    )
    return "web_search_agent"


def route_by_self_rag(state: GraphState) -> str:
    """Select the next node based on Self-RAG grounding verdict.

    Routes to END on PASS, or when the self_rag retry limit is exhausted.
    Routes back to generation_agent for one more attempt if not grounded
    and retries remain.

    Args:
        state: Current graph state.

    Returns:
        ``\"end\"`` to terminate, ``\"generation_agent\"`` to retry generation.
    """
    grounded = state.get("grounded", True)
    self_rag_retry_count = state.get("self_rag_retry_count", 0)
    logger.debug(
        "Self-RAG routing — grounded=%s, self_rag_retry_count=%d / MAX=%d",
        grounded, self_rag_retry_count, MAX_RETRIES,
    )

    if grounded:
        logger.info("Self-RAG PASS → END")
        return "end"

    if self_rag_retry_count < MAX_RETRIES:
        logger.info(
            "Self-RAG FAIL — retrying generation (%d/%d)", self_rag_retry_count, MAX_RETRIES
        )
        return "generation_agent"

    logger.warning(
        "Self-RAG retry limit reached (%d/%d) → END", self_rag_retry_count, MAX_RETRIES
    )
    return "end"
