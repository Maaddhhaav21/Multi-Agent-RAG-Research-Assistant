"""
local_llm_client.py
--------------------
Client for a locally-running model via Ollama. This is what makes your
"multi-model" story genuinely interesting for a portfolio project: you're
not just calling the same provider twice, you're comparing a frontier API
model against a free, locally-hosted open-weight model and letting the
fusion node judge between them.

Prerequisite (on your own machine, not in this sandbox):
    1. Install Ollama: https://ollama.com
    2. Run: ollama pull llama3
    3. Ollama runs a local server at http://localhost:11434 automatically

No API key needed -- that's the point of including this.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


def call_local_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    model: the Ollama model tag, e.g. "llama3" or "mistral"
    (must match what you've pulled locally via `ollama pull <model>`)
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]