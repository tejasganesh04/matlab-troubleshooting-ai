"""LangGraph workflow builder for the MATLAB Troubleshooting AI.

Assembles the StateGraph by registering nodes and conditional edges,
then compiles it into an executable application object. This is the
only module that has knowledge of the full graph topology.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from graph.edges import (
    route_after_crag,
    route_by_grading,
    route_by_intent,
    route_by_self_rag,
)
from graph.nodes import (
    run_crag_agent,
    run_generation_agent,
    run_grading_agent,
    run_intent_agent,
    run_retrieval_agent,
    run_self_rag_agent,
    run_vision_agent,
    run_web_search_agent,
)
from graph.state import GraphState

logger: logging.Logger = logging.getLogger(__name__)


def build_graph():
    """Build and compile the MATLAB Troubleshooting LangGraph workflow.

    Graph topology
    --------------
    intent_agent
        ├─ GENERAL ───────────────────► generation_agent
        ├─ IMAGE ─────────────────────► vision_agent ──► web_search_agent ──► generation_agent
        ├─ WEB_SEARCH ────────────────► web_search_agent ──► generation_agent
        └─ RAG ───────────────────────► retrieval_agent
                                             └──► grading_agent
                                                      ├─ GOOD ──► web_search_agent ──► generation_agent
                                                      └─ BAD ───► crag_agent
                                                                      ├─ retry < MAX ──► retrieval_agent
                                                                      └─ exhausted ────► web_search_agent ──► generation_agent
    generation_agent ──► self_rag_agent
                              ├─ PASS ────────────────────────────────────► END
                              ├─ FAIL + self_rag_retry < MAX ──► generation_agent
                              └─ FAIL + exhausted ────────────────────────► END

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``.invoke()`` or
        ``.stream()`` calls.
    """
    workflow: StateGraph = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("intent_agent",     run_intent_agent)
    workflow.add_node("vision_agent",     run_vision_agent)
    workflow.add_node("retrieval_agent",  run_retrieval_agent)
    workflow.add_node("grading_agent",    run_grading_agent)
    workflow.add_node("crag_agent",       run_crag_agent)
    workflow.add_node("web_search_agent", run_web_search_agent)
    workflow.add_node("generation_agent", run_generation_agent)
    workflow.add_node("self_rag_agent",   run_self_rag_agent)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("intent_agent")

    # ── Intent routing ────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "intent_agent",
        route_by_intent,
        {
            "vision_agent":     "vision_agent",
            "retrieval_agent":  "retrieval_agent",
            "web_search_agent": "web_search_agent",
            "generation_agent": "generation_agent",
        },
    )

    # ── Vision → Web Search → Generation ─────────────────────────────────────
    workflow.add_edge("vision_agent", "web_search_agent")

    # ── Retrieval → Grading ───────────────────────────────────────────────────
    workflow.add_edge("retrieval_agent", "grading_agent")

    # ── Grading routing ───────────────────────────────────────────────────────
    # GOOD → web_search_agent (enrich with web then generate)
    # BAD  → crag_agent (rewrite query and retry retrieval)
    workflow.add_conditional_edges(
        "grading_agent",
        route_by_grading,
        {
            "web_search_agent": "web_search_agent",
            "crag_agent":       "crag_agent",
        },
    )

    # ── CRAG routing ──────────────────────────────────────────────────────────
    # retry < MAX → retrieval_agent
    # exhausted   → web_search_agent (still run web search before generating)
    workflow.add_conditional_edges(
        "crag_agent",
        route_after_crag,
        {
            "retrieval_agent":  "retrieval_agent",
            "web_search_agent": "web_search_agent",
        },
    )

    # ── Web Search → Generation ───────────────────────────────────────────────
    workflow.add_edge("web_search_agent", "generation_agent")

    # ── Generation → Self-RAG ─────────────────────────────────────────────────
    workflow.add_edge("generation_agent", "self_rag_agent")

    # ── Self-RAG routing ──────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "self_rag_agent",
        route_by_self_rag,
        {
            "generation_agent": "generation_agent",
            "end":              END,
        },
    )

    app = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return app
