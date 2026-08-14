"""
builder.py
----------
This is the file that turns everything you've built so far into ONE runnable
graph. Nothing before this phase could actually execute end-to-end --
router.py, retrievers.py, generators.py, and fusion.py were all independently
correct, but nothing connected them.

The shape of the graph:

    START
      |
    router                              (decides which domains matter)
      |
      +-- (conditional, one branch per selected domain) --+
      |                    |                    |          |
    retrieve_code_docs  retrieve_finance  retrieve_general_kb  retrieve_uploaded_doc
      |                    |                    |          |
      +--> generate_*_gpt4o_mini    (each retriever feeds its own 2 generators)
      +--> generate_*_llama3
      |                    |                    |          |
      +----------------------- fusion -----------------------+
                             |
                            END

Key LangGraph concept used here: a node with MULTIPLE incoming edges
automatically waits for ALL of them to complete before running. That's how
fusion correctly waits for every generator node that got triggered, without
you writing any manual synchronization code.
"""

from langgraph.graph import StateGraph, START, END

from app.graph.state import GraphState, create_initial_state
from app.graph.router import route_query, route_to_retrievers
from app.retrieval.retrievers import (
    retrieve_code_docs,
    retrieve_finance,
    retrieve_general_kb,
    retrieve_uploaded_doc,
)
from app.graph.generators import (
    generate_code_docs_gpt4o_mini,
    generate_code_docs_llama3,
    generate_finance_gpt4o_mini,
    generate_finance_llama3,
    generate_general_kb_gpt4o_mini,
    generate_general_kb_llama3,
    generate_uploaded_doc_gpt4o_mini,
    generate_uploaded_doc_llama3,
)
from app.graph.fusion import fuse_candidates


def build_graph():
    graph = StateGraph(GraphState)

    # --- Register every node ---
    graph.add_node("router", route_query)

    graph.add_node("retrieve_code_docs", retrieve_code_docs)
    graph.add_node("retrieve_finance", retrieve_finance)
    graph.add_node("retrieve_general_kb", retrieve_general_kb)
    graph.add_node("retrieve_uploaded_doc", retrieve_uploaded_doc)

    graph.add_node("generate_code_docs_gpt4o_mini", generate_code_docs_gpt4o_mini)
    graph.add_node("generate_code_docs_llama3", generate_code_docs_llama3)
    graph.add_node("generate_finance_gpt4o_mini", generate_finance_gpt4o_mini)
    graph.add_node("generate_finance_llama3", generate_finance_llama3)
    graph.add_node("generate_general_kb_gpt4o_mini", generate_general_kb_gpt4o_mini)
    graph.add_node("generate_general_kb_llama3", generate_general_kb_llama3)
    graph.add_node("generate_uploaded_doc_gpt4o_mini", generate_uploaded_doc_gpt4o_mini)
    graph.add_node("generate_uploaded_doc_llama3", generate_uploaded_doc_llama3)

    graph.add_node("fusion", fuse_candidates)

    # --- Wire the edges ---
    graph.add_edge(START, "router")

    # Conditional fan-out: router decides WHICH of these run. If selected_domains
    # is e.g. ["finance"], only retrieve_finance executes -- the others simply
    # never run for this query, which is exactly why fusion only waits for
    # whichever generator nodes actually fired.
    graph.add_conditional_edges(
        "router",
        route_to_retrievers,
        {
            "code_docs": "retrieve_code_docs",
            "finance": "retrieve_finance",
            "general_kb": "retrieve_general_kb",
            "uploaded_doc": "retrieve_uploaded_doc",
        },
    )

    # Each retriever unconditionally feeds its own two generator nodes.
    # These are plain edges (not conditional) because if the retriever ran
    # at all, we always want both models to generate a candidate from it.
    graph.add_edge("retrieve_code_docs", "generate_code_docs_gpt4o_mini")
    graph.add_edge("retrieve_code_docs", "generate_code_docs_llama3")

    graph.add_edge("retrieve_finance", "generate_finance_gpt4o_mini")
    graph.add_edge("retrieve_finance", "generate_finance_llama3")

    graph.add_edge("retrieve_general_kb", "generate_general_kb_gpt4o_mini")
    graph.add_edge("retrieve_general_kb", "generate_general_kb_llama3")

    graph.add_edge("retrieve_uploaded_doc", "generate_uploaded_doc_gpt4o_mini")
    graph.add_edge("retrieve_uploaded_doc", "generate_uploaded_doc_llama3")

    # Fan-in: every generator node points to fusion. LangGraph only waits for
    # the ones that are actually reachable given this run's conditional
    # routing -- an unselected domain's generators never entered the
    # execution path, so fusion doesn't wait on them.
    for node_name in [
        "generate_code_docs_gpt4o_mini", "generate_code_docs_llama3",
        "generate_finance_gpt4o_mini", "generate_finance_llama3",
        "generate_general_kb_gpt4o_mini", "generate_general_kb_llama3",
        "generate_uploaded_doc_gpt4o_mini", "generate_uploaded_doc_llama3",
    ]:
        graph.add_edge(node_name, "fusion")

    graph.add_edge("fusion", END)

    return graph.compile()


# Compiled once at import time so app/main.py (or tests) can just do:
#   from app.graph.builder import compiled_graph
#   result = compiled_graph.invoke(create_initial_state(...))
compiled_graph = build_graph()