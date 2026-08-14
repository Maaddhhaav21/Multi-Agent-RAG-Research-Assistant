"""
generators.py
--------------
Generator nodes. Each one takes the chunks a retriever already pulled for
ONE domain, sends them to ONE model, and produces a CandidateAnswer.

This is where the "multi-agent" part of the project actually happens: instead
of one generation step, you get one candidate answer per (domain, model)
combination that's configured to run. The fusion node (Phase 4) then picks
the best answer or synthesizes across all of them.

Design choice: each generator node is scoped to a SINGLE domain + SINGLE
model, not "all domains, all models" in one function. This keeps every node
small, independently testable, and lets LangGraph run them all in parallel
(they don't depend on each other's output).
"""

from typing import Dict, Callable
from app.graph.state import GraphState, Domain, CandidateAnswer
from app.models.openai_client import call_openai
from app.models.local_llm_client import call_local_llm


GENERATION_SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided context.

Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so explicitly
  rather than guessing.
- Be concise and direct.
"""


def _build_user_prompt(query: str, context_chunks: list[str]) -> str:
    context_block = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
    return f"Context:\n{context_block}\n\nQuestion: {query}"


def make_generator_node(domain: Domain, model_name: str, call_fn: Callable[[str, str, str], str]):
    """
    Factory that produces a generator node for one (domain, model) pair.

    call_fn: the underlying client function to invoke, e.g. call_openai or
    call_local_llm. Both share the same signature (model, system, user) ->
    str, which is what makes this factory work uniformly across providers --
    the generator logic doesn't care WHICH provider it's calling, only that
    it conforms to that shape.
    """

    def generate(state: GraphState) -> Dict:
        # If this domain wasn't selected by the router, or retrieval came
        # back empty, skip generation entirely rather than sending an LLM
        # call with no context (wastes money and produces a useless answer).
        docs = state["retrieved_docs"].get(domain, [])
        if not docs:
            return {"candidate_answers": []}

        chunks = [d.page_content for d in docs]
        user_prompt = _build_user_prompt(state["query"], chunks)

        answer_text = call_fn(model_name, GENERATION_SYSTEM_PROMPT, user_prompt)

        candidate: CandidateAnswer = {
            "domain": domain,
            "model_name": model_name,
            "answer_text": answer_text,
            "supporting_chunks": chunks,
        }
        # Wrapped in a list because candidate_answers uses operator.add as its
        # reducer (see state.py) -- each parallel node contributes its own
        # single-item list, and LangGraph concatenates them all together.
        return {"candidate_answers": [candidate]}

    generate.__name__ = f"generate_{domain}_{model_name}".replace("-", "_").replace(".", "_")
    return generate


# ---------------------------------------------------------------------------
# Concrete generator nodes.
#
# This is where you decide your model lineup. Example below: each domain
# gets answered by TWO models (gpt-4o-mini and a local llama3), so the
# fusion node has something real to compare. Add/remove pairs freely --
# builder.py just needs to know which node names exist.
# ---------------------------------------------------------------------------

generate_code_docs_gpt4o_mini = make_generator_node("code_docs", "gpt-4o-mini", call_openai)
generate_code_docs_llama3 = make_generator_node("code_docs", "llama3", call_local_llm)

generate_finance_gpt4o_mini = make_generator_node("finance", "gpt-4o-mini", call_openai)
generate_finance_llama3 = make_generator_node("finance", "llama3", call_local_llm)

generate_general_kb_gpt4o_mini = make_generator_node("general_kb", "gpt-4o-mini", call_openai)
generate_general_kb_llama3 = make_generator_node("general_kb", "llama3", call_local_llm)

generate_uploaded_doc_gpt4o_mini = make_generator_node("uploaded_doc", "gpt-4o-mini", call_openai)
generate_uploaded_doc_llama3 = make_generator_node("uploaded_doc", "llama3", call_local_llm)