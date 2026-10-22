"""Typed state definition for the MATLAB Troubleshooting LangGraph workflow.

All LangGraph nodes read from and write to this shared state object.
No agent or node should carry its own mutable state outside of GraphState.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """Immutable-by-convention shared state passed between all graph nodes.

    Attributes:
        user_query: The original text question submitted by the user.
        uploaded_image: Raw bytes of the uploaded image, or None if absent.
        intent: Classified intent — one of GENERAL, RAG, IMAGE, WEB_SEARCH.
        rewritten_query: Query rewritten by the CRAG agent for better retrieval.
        retrieved_documents: List of raw document strings from the retriever.
        retrieved_context: Concatenated retrieval context passed to the generator.
        vision_context: Structured text extracted by the Vision Agent.
        web_context: Text returned by the Web Search Agent.
        final_context: Assembled context combining all available sources.
        generated_answer: The final answer produced by the Generation Agent.
        grounded: True if Self-RAG verified the answer is grounded, else False.
        crag_retry_count: Counter incremented by the CRAG agent only (max 2).
        self_rag_retry_count: Counter incremented by Self-RAG on failure (max 2).
        retrieval_grade: Grade assigned by the Retrieval Grading Agent: GOOD or BAD.
    """

    user_query: str
    uploaded_image: Optional[bytes]
    intent: str
    rewritten_query: str
    retrieved_documents: list[str]
    retrieved_context: str
    vision_context: str
    web_context: str
    final_context: str
    generated_answer: str
    grounded: bool
    crag_retry_count: int
    self_rag_retry_count: int
    retrieval_grade: str


MAX_RETRIES: int = 2
