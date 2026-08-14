"""
router_llm_client.py
---------------------
Placeholder client for Phase 1 so router.py can actually run end-to-end.

In Phase 2 you'll build out full model clients (claude_client.py, openai_client.py,
local_llm_client.py) for GENERATION. This file stays separate and stays simple
on purpose: routing should always use your fastest/cheapest model, so it makes
sense to keep its client minimal rather than reusing your heavyweight generation
clients.

Uses OpenAI here -- specifically gpt-4o-mini, which is OpenAI's cheap/fast tier,
for the same reason we'd pick Haiku on the Anthropic side: routing is a simple
classification task and doesn't need a frontier model.
"""

import os
from functools import lru_cache
from openai import OpenAI


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """
    Lazy client creation -- see openai_client.py (Phase 3) for why this
    matters: instantiating OpenAI() at import time crashes any code path
    that imports this module before OPENAI_API_KEY is set, even if that
    code path never actually calls the API.
    """
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def call_router_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Calls a cheap/fast model purely for classification.
    Using gpt-4o-mini here keeps routing latency and cost low --
    you don't want your router to be the slowest part of the pipeline.
    """
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content