"""
router.py
---------
The FIRST node the query hits after entering the graph.

Job: look at the user's query (and whether they've uploaded a PDF this session)
and decide which domain(s) are worth retrieving from. This output directly
controls which retriever nodes LangGraph will execute next.

We implement classification via an LLM call with a constrained JSON output,
rather than keyword matching, because query intent isn't always obvious from
keywords alone (e.g. "what's the exposure here" could be finance or the
uploaded doc, depending on context).
"""

import json
from typing import List
from app.graph.state import GraphState, Domain

# NOTE: this import will exist once you build Phase 2 (models/). For now it's
# a thin wrapper you'll create around whichever LLM you use for routing.
# Routing should use a FAST, CHEAP model -- you don't need your strongest
# model to classify a query into a handful of categories.
from app.models.router_llm_client import call_router_llm


ALL_STATIC_DOMAINS: List[Domain] = ["code_docs", "finance", "general_kb"]

ROUTER_SYSTEM_PROMPT = """You are a query router for a multi-domain RAG system.

Given a user query, decide which knowledge domain(s) are relevant. Available domains:
- code_docs: programming, APIs, technical documentation questions
- finance: market data, financial news, economic questions
- general_kb: general knowledge questions not covered by the above
- uploaded_doc: only include this if explicitly told the user has an uploaded document
  AND the query could plausibly relate to it

Respond with ONLY a JSON array of domain strings, nothing else.
Example: ["finance"]
Example: ["uploaded_doc", "finance"]

If genuinely unsure, include general_kb as a fallback rather than guessing wrong.
"""


def _build_user_prompt(query: str, has_uploaded_doc: bool) -> str:
    doc_note = (
        "The user HAS an uploaded document active in this session."
        if has_uploaded_doc
        else "The user has NOT uploaded any document in this session "
             "(do not select uploaded_doc)."
    )
    return f"{doc_note}\n\nQuery: {query}"


def _parse_domains(raw_response: str) -> List[Domain]:
    """
    LLMs are unreliable about returning ONLY JSON, even when told to.
    This defensively extracts the JSON array and falls back safely if parsing fails.
    """
    try:
        # Strip common wrapping like ```json ... ``` if the model adds it anyway
        cleaned = raw_response.strip().strip("`").replace("json\n", "", 1)
        domains = json.loads(cleaned)

        if not isinstance(domains, list) or not domains:
            raise ValueError("Router returned empty or non-list output")

        # Only keep values that are actually valid Domain options -- never trust
        # raw LLM output to already be safe/clean.
        valid = {"code_docs", "finance", "general_kb", "uploaded_doc"}
        filtered = [d for d in domains if d in valid]

        return filtered if filtered else ["general_kb"]

    except (json.JSONDecodeError, ValueError):
        # Fail safe: if we can't parse the router's decision, default to the
        # general knowledge base rather than crashing the whole graph.
        return ["general_kb"]


def route_query(state: GraphState) -> dict:
    """
    LangGraph node function.

    IMPORTANT LangGraph convention: a node function takes the full state and
    returns a PARTIAL dict of only the fields it's updating. LangGraph merges
    this back into the main state automatically -- you never mutate `state`
    directly.
    """
    query = state["query"]
    has_uploaded_doc = state["has_uploaded_doc"]

    raw_response = call_router_llm(
        system_prompt=ROUTER_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(query, has_uploaded_doc),
    )

    domains = _parse_domains(raw_response)

    # Safety net: never let the router select uploaded_doc if there isn't one,
    # even if the LLM hallucinates it. Deterministic code should always be the
    # final gate on top of LLM decisions for anything that affects control flow.
    if not has_uploaded_doc and "uploaded_doc" in domains:
        domains.remove("uploaded_doc")
        if not domains:
            domains = ["general_kb"]

    return {"selected_domains": domains}


def route_to_retrievers(state: GraphState) -> List[str]:
    """
    This is a CONDITIONAL EDGE function, not a node. LangGraph calls this
    after route_query() runs to decide which retriever node(s) to fan out to
    next. It returns node NAMES (strings), which must match how you register
    them in builder.py (Phase 1 continuation) e.g.:

        graph.add_conditional_edges("router", route_to_retrievers, {
            "code_docs": "retrieve_code_docs",
            "finance": "retrieve_finance",
            "general_kb": "retrieve_general_kb",
            "uploaded_doc": "retrieve_uploaded_doc",
        })

    Returning a list (not a single string) is what makes LangGraph run the
    matched retriever nodes in PARALLEL rather than one after another.
    """
    return state["selected_domains"]