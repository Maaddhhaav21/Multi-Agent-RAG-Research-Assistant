"""
vector_store.py
----------------
Thin wrapper around the Qdrant client. Handles both:
1. Permanent collections (code_docs, finance, general_kb) -- built once via
   scripts/build_indexes.py and reused across all sessions.
2. Session-scoped collections (one per uploaded PDF) -- created on upload,
   deleted when the session ends.

Keeping this as a wrapper (instead of calling qdrant_client directly all over
the codebase) means if you ever swap Qdrant for another vector DB, you only
change this one file.
"""

import uuid
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document

from app.retrieval.embeddings import EMBEDDING_DIM

# Local/dev default. Swap for a hosted Qdrant URL + API key in production
# via env vars -- keeping it hardcoded here for Phase 2 simplicity.
_client = QdrantClient(path="./data/qdrant_local")

# Static domain collections are created once. Session collections use a
# naming convention (see session_collection_name) so we can find + delete
# them without keeping a separate registry.
STATIC_COLLECTIONS = ["code_docs", "finance", "general_kb"]


def session_collection_name(session_id: str) -> str:
    """
    Deterministic naming so any part of the code can derive the collection
    name from a session_id without a lookup table.
    """
    return f"session_{session_id}"


def ensure_collection(collection_name: str) -> None:
    """
    Creates the collection if it doesn't already exist. Safe to call
    repeatedly -- checks existence first rather than relying on try/except,
    since Qdrant's "already exists" error handling varies by version.
    """
    existing = {c.name for c in _client.get_collections().collections}
    if collection_name not in existing:
        _client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def upsert_documents(
    collection_name: str,
    chunks: List[str],
    vectors: List[List[float]],
    metadata: Optional[List[Dict]] = None,
) -> None:
    """
    Writes chunks + their embeddings into a collection.
    `metadata` (e.g. {"source": "readme.md", "page": 3}) travels alongside
    each vector so retrieved results can be traced back to their origin --
    important for the citations field in candidate_answers later.
    """
    ensure_collection(collection_name)
    metadata = metadata or [{} for _ in chunks]

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, **meta},
        )
        for chunk, vector, meta in zip(chunks, vectors, metadata)
    ]
    _client.upsert(collection_name=collection_name, points=points)


def search(collection_name: str, query_vector: List[float], top_k: int = 5) -> List[Document]:
    """
    Returns the top_k most similar chunks as LangChain Document objects, so
    the rest of the graph (generators, fusion) can work with a consistent
    type regardless of which vector store or domain the chunks came from.
    """
    existing = {c.name for c in _client.get_collections().collections}
    if collection_name not in existing:
        # A domain with no data yet (e.g. no PDF uploaded) should return
        # nothing gracefully, not crash the graph.
        return []

    # NOTE: QdrantClient.search() was deprecated in favor of query_points()
    # as of qdrant-client 1.10+. query_points() wraps results in a
    # `.points` attribute instead of returning the list directly.
    response = _client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    return [
        Document(page_content=hit.payload.get("text", ""), metadata=hit.payload)
        for hit in response.points
    ]


def delete_session_collection(session_id: str) -> None:
    """
    Cleanup for when a session ends. Call this from session_store.py's
    TTL/cleanup logic -- without it, every uploaded PDF leaves a permanent
    orphaned collection in Qdrant.
    """
    name = session_collection_name(session_id)
    existing = {c.name for c in _client.get_collections().collections}
    if name in existing:
        _client.delete_collection(collection_name=name)