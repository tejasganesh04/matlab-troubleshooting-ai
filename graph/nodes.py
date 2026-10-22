"""LangGraph node functions for the MATLAB Troubleshooting AI workflow.

Each function here is a LangGraph node. Nodes receive the full GraphState,
call the appropriate agent function, and return a partial state dict
containing only the keys that were updated.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.crag_agent import rewrite_query
from agents.generation_agent import generate_answer
from agents.grading_agent import grade_retrieval
from agents.intent_agent import classify_intent
from agents.retrieval_agent import retrieve_documents
from agents.self_rag_agent import verify_answer
from agents.vision_agent import analyze_image
from agents.web_search_agent import search_web
from graph.state import GraphState

logger: logging.Logger = logging.getLogger(__name__)


def run_intent_agent(state: GraphState) -> dict[str, Any]:
    """Classify the user query intent.

    Args:
        state: Current graph state.

    Returns:
        Partial state update: {\"intent\": str}.
    """
    logger.info("Node: intent_agent")

    intent = classify_intent(
        user_query=state["user_query"],
        has_image=state.get("uploaded_image") is not None,
    )

    print(f"\n========== INTENT ==========")
    print(intent)
    print("============================\n")

    return {"intent": intent}


def run_vision_agent(state: GraphState) -> dict[str, Any]:
    """Analyse the uploaded image and write extracted text to state.

    Args:
        state: Current graph state.

    Returns:
        Partial state update: {\"vision_context\": str}.
    """
    logger.info("Node: vision_agent")
    image_bytes: bytes | None = state.get("uploaded_image")

    if not image_bytes:
        logger.warning("vision_agent called with no image in state.")
        return {"vision_context": ""}

    vision_text = analyze_image(image_bytes)
    return {"vision_context": vision_text}


def run_retrieval_agent(state: GraphState) -> dict[str, Any]:
    """Retrieve relevant documents from the FAISS knowledge base.

    Args:
        state: Current graph state.

    Returns:
        Partial state update: {\"retrieved_documents\": list, \"retrieved_context\": str}.
    """
    logger.info("Node: retrieval_agent")

    query = state.get("rewritten_query") or state["user_query"]

    print("\n========== RETRIEVAL QUERY ==========")
    print(query)
    print("=====================================\n")

    documents, context = retrieve_documents(query)

    print("\n========== RETRIEVED CONTEXT ==========")
    print(context[:500] if context else "(empty)")
    print("=======================================\n")

    return {
        "retrieved_documents": documents,
        "retrieved_context": context,
    }


def run_grading_agent(state: GraphState) -> dict[str, Any]:
    """Grade the quality of the retrieved context.

    Args:
        state: Current graph state.

    Returns:
        Partial state update: {\"retrieval_grade\": str}.
    """
    logger.info("Node: grading_agent")
    grade = grade_retrieval(
        query=state["user_query"],
        context=state.get("retrieved_context", ""),
    )
    return {"retrieval_grade": grade}


def run_crag_agent(state: GraphState) -> dict[str, Any]:
    """Rewrite the query and increment the CRAG retry counter.

    Args:
        state: Current graph state.

    Returns:
        Partial state update: {\"rewritten_query\": str, \"crag_retry_count\": int}.
    """
    logger.info("Node: crag_agent")
    current_query = state.get("rewritten_query") or state["user_query"]
    rewritten = rewrite_query(current_query)
    crag_retry_count = state.get("crag_retry_count", 0) + 1
    logger.info("CRAG crag_retry_count → %d", crag_retry_count)
    return {"rewritten_query": rewritten, "crag_retry_count": crag_retry_count}


def run_web_search_agent(state: GraphState) -> dict[str, Any]:
    """Search the web for additional MATLAB-related information.

    Args:
        state: Current graph state.

    Returns:
        Partial state update: {\"web_context\": str}.
    """
    logger.info("Node: web_search_agent")

    query = state.get("rewritten_query") or state["user_query"]

    web_context = search_web(query)

    return {
        "web_context": web_context,
    }


def run_generation_agent(state: GraphState) -> dict[str, Any]:
    """Generate the final answer and assemble the combined final_context.

    Args:
        state: Current graph state.

    Returns:
        Partial state update: {\"generated_answer\": str, \"final_context\": str}.
    """
    logger.info("Node: generation_agent")
    retrieved_context = state.get("retrieved_context", "")
    vision_context = state.get("vision_context", "")
    web_context = state.get("web_context", "")

    query = state.get("rewritten_query") or state["user_query"]

    print("\n========== CONTEXT SENT TO LLM ==========")
    print(retrieved_context[:500] if retrieved_context else "(empty)")
    print("=========================================\n")

    answer = generate_answer(
        query=query,
        retrieved_context=retrieved_context,
        vision_context=vision_context,
        web_context=web_context,
    )

    final_context = "\n\n".join(
        c.strip() for c in [vision_context, retrieved_context, web_context] if c.strip()
    )

    return {"generated_answer": answer, "final_context": final_context}


def run_self_rag_agent(state: GraphState) -> dict[str, Any]:
    """Verify answer grounding and increment self_rag_retry_count on failure.

    Args:
        state: Current graph state.

    Returns:
        Partial state update: {\"grounded\": bool, \"self_rag_retry_count\": int}.
    """
    logger.info("Node: self_rag_agent")
    verdict = verify_answer(
        query=state["user_query"],
        context=state.get("final_context", ""),
        answer=state.get("generated_answer", ""),
    )
    grounded = verdict == "PASS"
    self_rag_retry_count = state.get("self_rag_retry_count", 0)

    if not grounded:
        self_rag_retry_count += 1
        logger.info("Self-RAG FAIL — self_rag_retry_count → %d", self_rag_retry_count)

    return {"grounded": grounded, "self_rag_retry_count": self_rag_retry_count}
