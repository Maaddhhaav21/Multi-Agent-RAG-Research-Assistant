"""
retrievers.py
-------------
Retriever nodes. Each one runs after the router's conditional edge selects it,
and each writes ONLY to its own key in `retrieved_docs` (see state.py) so
parallel execution never causes a race condition between domains.

Note the pattern: instead of writing 4 nearly-identical functions, we use a
factory (`make_retriever_node`) that generates a node function per domain.
This keeps the logic in one place -- if you change how retrieval works, you
change it once, not 4 times.
"""

from typing import Dict
from app.graph.state import GraphState, Domain
from app.retrieval.embeddings import embed_query
from app.retrieval.vector_store import search, session_collection_name


def make_retriever_node(domain: Domain):
    """
    Returns a LangGraph node function scoped to one domain.

    Why a closure instead of one big if/elif function? Because LangGraph
    registers nodes by name (see builder.py), and each node needs to be a
    distinct callable. A closure lets us parameterize by domain while still
    producing separate, independently-testable functions.
    """

    def retrieve(state: GraphState) -> Dict:
        query = state["query"]
        query_vector = embed_query(query)

        # The uploaded_doc domain uses a session-specific collection name;
        # every other domain uses its fixed, permanent collection name.
        collection_name = (
            session_collection_name(state["session_id"])
            if domain == "uploaded_doc"
            else domain
        )

        docs = search(collection_name=collection_name, query_vector=query_vector, top_k=5)

        # We return a dict with the domain as the key inside retrieved_docs,
        # NOT the whole retrieved_docs dict -- LangGraph needs to know how to
        # merge partial updates from parallel nodes. See builder.py's reducer
        # config for how retrieved_docs specifically merges (Phase 1 note:
        # you'll add `Annotated[Dict, merge_dicts]` there once you wire this up).
        return {"retrieved_docs": {domain: docs}}

    # Helpful for debugging graph execution logs -- without this, every
    # retriever function shows up as "retrieve" in LangGraph's trace output.
    retrieve.__name__ = f"retrieve_{domain}"
    return retrieve


# One node per domain, built from the same factory.
retrieve_code_docs = make_retriever_node("code_docs")
retrieve_finance = make_retriever_node("finance")
retrieve_general_kb = make_retriever_node("general_kb")
retrieve_uploaded_doc = make_retriever_node("uploaded_doc")