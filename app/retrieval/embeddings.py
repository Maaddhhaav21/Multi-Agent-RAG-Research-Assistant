"""
embeddings.py
-------------
Wraps whichever embedding model we use, so the rest of the codebase never
imports sentence-transformers (or OpenAI embeddings) directly.

Why this matters: embeddings are used in TWO places -- once when you build the
static indexes (scripts/build_indexes.py) and once when you embed a user's
uploaded PDF on the fly (retrieval/session/session_indexer.py). Both places
MUST use the exact same embedding model, or the vectors won't be comparable
and retrieval quality silently degrades. Centralizing it here means there's
only one place that can get out of sync.

We use a local sentence-transformers model (bge-large) instead of an API-based
embedding model. Two reasons:
1. Cost -- you'll be re-embedding uploaded PDFs constantly while testing.
2. Latency -- no network round trip means faster iteration during development.
"""

from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024  # must match the vector size configured in vector_store.py


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Cached loader -- the model is ~1.3GB and slow to load. lru_cache ensures
    it's loaded once per process, not once per function call.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch-embeds a list of text chunks. Always prefer this over embedding one
    chunk at a time in a loop -- batching is significantly faster on both CPU
    and GPU.
    """
    model = _get_model()
    # normalize_embeddings=True makes cosine similarity == dot product,
    # which is what Qdrant's default distance metric expects for this model.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """
    Single-query convenience wrapper. bge models expect a specific instruction
    prefix on QUERIES (but not on documents) for best retrieval performance --
    this is a quirk of how bge was trained, not a general embedding rule.
    """
    instruction = "Represent this sentence for searching relevant passages: "
    return embed_texts([instruction + query])[0]