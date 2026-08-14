"""
openai_client.py
-----------------
Generation client for OpenAI models. Kept separate from router_llm_client.py
(Phase 1) even though both use OpenAI, because these serve different purposes:
this one answers questions using retrieved context (needs a capable model),
routing just classifies into categories (cheap model is fine).

Exposes ONE function that takes a model name as a parameter, rather than one
function per model -- this lets generators.py request "gpt-4o-mini" or
"gpt-4o" through the same interface, which matters once you build the
factory pattern in generators.py.
"""

import os
from functools import lru_cache
from openai import OpenAI


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """
    Lazy client creation. If we instantiated OpenAI() at module import time
    instead, simply IMPORTING this file (e.g. via generators.py) would crash
    with "Missing credentials" the moment OPENAI_API_KEY isn't set yet --
    even in code paths that never actually call the API (like running tests
    with mocked functions). Creating it on first real use avoids that.
    """
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def call_openai(model: str, system_prompt: str, user_prompt: str) -> str:
    response = _get_client().chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content