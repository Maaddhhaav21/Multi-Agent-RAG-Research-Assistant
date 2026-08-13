"""
state.py
--------
Defines the shared state object that flows through every node in the LangGraph.

In LangGraph, the "state" is a single dict-like object that every node reads from
and writes back to. Instead of passing separate arguments between functions (like
in a normal Python program), every node gets the FULL state, and returns a partial
update that LangGraph merges back in.

This is the most important file to get right early, because every node you write
later depends on this schema. Changing it after you've built 5 nodes means
touching all 5 nodes again.
"""

from typing import TypedDict, List, Dict, Optional, Literal
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Sub-types: small, well-defined pieces used inside the main state
# ---------------------------------------------------------------------------

# The set of domains our router can send a query to.
# "uploaded_doc" only becomes valid if the user has uploaded a PDF this session.
Domain = Literal["code_docs", "finance", "general_kb", "uploaded_doc"]


class CandidateAnswer(TypedDict):
    """
    One model's answer to the query, grounded in one domain's retrieved context.
    We keep these separate (instead of immediately merging) so the fusion node
    can compare them side by side later.
    """
    domain: Domain
    model_name: str          # e.g. "claude-sonnet", "gpt-4o", "llama3-local"
    answer_text: str
    supporting_chunks: List[str]   # raw text of the chunks used, for traceability


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    # --- Input ---
    query: str                          # the user's raw question
    session_id: str                     # ties this run to a specific chat session
    has_uploaded_doc: bool              # True if the session has an active uploaded PDF

    # --- Router output ---
    # The router decides which domain(s) are relevant. It's a list because a query
    # can legitimately span more than one domain (e.g. "compare this PDF's numbers
    # to general market trends" -> ["uploaded_doc", "finance"]).
    selected_domains: List[Domain]

    # --- Retrieval output ---
    # Keyed by domain, so each retriever node only writes to its own key and
    # never clobbers another domain's results.
    retrieved_docs: Dict[Domain, List[Document]]

    # --- Generation output ---
    # One candidate answer per (domain, model) pair that ran.
    candidate_answers: List[CandidateAnswer]

    # --- Fusion output ---
    final_answer: Optional[str]
    final_confidence: Optional[float]   # 0.0-1.0, set by the fusion/judge node
    citations: List[str]                # which domains/models contributed

    # --- Control flow ---
    # Used by the critique/re-retrieval loop (Phase 3). Capped so we never loop forever.
    retrieval_attempts: int


def create_initial_state(
    query: str,
    session_id: str,
    has_uploaded_doc: bool = False,
) -> GraphState:
    """
    Factory function for a fresh state at the start of every graph run.

    Why a factory function instead of building the dict inline in main.py?
    Because it guarantees every field is initialized consistently, and if you
    add a new field to GraphState later, you only update it in ONE place.
    """
    return GraphState(
        query=query,
        session_id=session_id,
        has_uploaded_doc=has_uploaded_doc,
        selected_domains=[],
        retrieved_docs={},
        candidate_answers=[],
        final_answer=None,
        final_confidence=None,
        citations=[],
        retrieval_attempts=0,
    )