"""
fusion.py
---------
The LAST node in the graph (for now -- Phase 5 adds an optional critique loop
before this). Takes every candidate answer produced by the generator nodes
and synthesizes ONE final answer, rather than just picking whichever
candidate happened to run first.

Why not just return the single "best" candidate as-is? Because different
candidates may be individually incomplete but collectively cover the full
answer -- e.g. the finance-domain answer covers "why rates rose" and the
uploaded-doc answer covers "what this means for your specific numbers."
A synthesis step can combine both instead of forcing an either/or choice.

This node also assigns a confidence score, which the critique loop (Phase 5)
will use to decide whether to trust the answer or re-retrieve with a
refined query.
"""

import json
from typing import Dict, List
from app.graph.state import GraphState, CandidateAnswer
from app.models.openai_client import call_openai

# Fusion is a REASONING task (comparing, resolving conflicts, synthesizing),
# not a simple classification like routing -- so unlike router_llm_client.py,
# this uses a stronger model rather than the cheap/fast tier.
FUSION_MODEL = "gpt-4o"

FUSION_SYSTEM_PROMPT = """You are a judge synthesizing multiple candidate answers into one final answer.

You will be given several candidate answers to the same question, each produced
by a different model using different retrieved context. Your job:
1. Identify where candidates agree -- that's likely reliable.
2. Identify contradictions -- flag them or resolve using the most specific/well-supported claim.
3. Combine complementary information from different candidates into one coherent answer.
4. Do NOT just repeat one candidate verbatim -- actually synthesize.
5. Assign a confidence score (0.0-1.0) reflecting how well-supported the final answer is
   by the retrieved context. Low confidence if candidates disagree heavily or context was thin.

Respond with ONLY valid JSON in this exact shape, nothing else:
{"final_answer": "...", "confidence": 0.0, "domains_used": ["..."]}
"""


def _format_candidates_for_prompt(candidates: List[CandidateAnswer]) -> str:
    blocks = []
    for i, c in enumerate(candidates, start=1):
        blocks.append(
            f"Candidate {i} (domain: {c['domain']}, model: {c['model_name']}):\n"
            f"{c['answer_text']}"
        )
    return "\n\n---\n\n".join(blocks)


def _parse_fusion_response(raw_response: str, candidates: List[CandidateAnswer]) -> Dict:
    """
    Defensive parsing, same principle as router.py's _parse_domains: never
    trust an LLM to return perfectly clean JSON, and always have a safe
    fallback so a parsing failure doesn't crash the whole graph run.
    """
    try:
        cleaned = raw_response.strip().strip("`").replace("json\n", "", 1)
        parsed = json.loads(cleaned)

        final_answer = parsed.get("final_answer", "").strip()
        confidence = float(parsed.get("confidence", 0.0))
        domains_used = parsed.get("domains_used", [])

        if not final_answer:
            raise ValueError("Empty final_answer from fusion model")

        # Clamp confidence into a valid range -- never trust a model to
        # respect numeric bounds you asked for in a prompt.
        confidence = max(0.0, min(1.0, confidence))

        return {
            "final_answer": final_answer,
            "final_confidence": confidence,
            "citations": domains_used,
        }

    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: if fusion parsing fails but we DO have candidates, surface
        # the first candidate's answer rather than returning nothing. Mark
        # confidence low so the critique loop (Phase 5) knows this wasn't a
        # clean synthesis.
        fallback_answer = candidates[0]["answer_text"] if candidates else (
            "I wasn't able to find enough information to answer that."
        )
        return {
            "final_answer": fallback_answer,
            "final_confidence": 0.2,
            "citations": [candidates[0]["domain"]] if candidates else [],
        }


def fuse_candidates(state: GraphState) -> Dict:
    """
    LangGraph node function. Runs once all active generator nodes have
    completed -- LangGraph automatically waits for every incoming edge into
    this node before executing it (see builder.py for how those edges are
    wired), so `state["candidate_answers"]` is guaranteed complete by the
    time this function runs.
    """
    candidates = state["candidate_answers"]

    if not candidates:
        # No domain matched, or every retriever came back empty. Don't call
        # an LLM with nothing to synthesize -- just return a clear message.
        return {
            "final_answer": "I couldn't find relevant information to answer that question.",
            "final_confidence": 0.0,
            "citations": [],
        }

    if len(candidates) == 1:
        # No real synthesis needed with only one candidate -- skip the LLM
        # call entirely and save the cost/latency.
        only = candidates[0]
        return {
            "final_answer": only["answer_text"],
            "final_confidence": 0.7,  # moderate default; single-source, unverified against alternatives
            "citations": [only["domain"]],
        }

    user_prompt = f"Question: {state['query']}\n\n{_format_candidates_for_prompt(candidates)}"
    raw_response = call_openai(FUSION_MODEL, FUSION_SYSTEM_PROMPT, user_prompt)

    return _parse_fusion_response(raw_response, candidates)