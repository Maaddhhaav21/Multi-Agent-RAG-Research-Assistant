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

from typing import TypedDict, List, Dict, Optional, Literal, Annotated
from langchain_core.documents import Document


def merge_retrieved_docs(
    existing: Dict[str, List[Document]],
    update: Dict[str, List[Document]],
) -> Dict[str, List[Document]]:
    """
    Custom reducer for retrieved_docs.

    LangGraph's DEFAULT merge behavior for a dict field is "last write wins" --
    if two retriever nodes finish in the same step and both return an update
    to retrieved_docs, whichever one LangGraph processes last overwrites the
    other's result entirely. Since our retriever nodes run in PARALLEL (one
    per selected domain), this would silently drop results.

    This reducer instead merges keys, so retrieve_finance's output and
    retrieve_uploaded_doc's output both survive even if they complete in the
    same graph step.
    """
    merged = dict(existing)
    merged.update(update)
    return merged


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
    # never clobbers another domain's results. Annotated with a custom reducer
    # (merge_retrieved_docs) because these writes happen in PARALLEL -- see
    # that function's docstring for why this matters.
    retrieved_docs: Annotated[Dict[Domain, List[Document]], merge_retrieved_docs]

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