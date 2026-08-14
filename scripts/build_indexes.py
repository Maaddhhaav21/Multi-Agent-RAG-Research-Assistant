"""
build_indexes.py
-----------------
Run this ONCE (and again any time you add/change files in data/raw/) to
populate your permanent Qdrant collections: code_docs, finance, general_kb.

This is the missing piece that connects "I have some files" to "my chatbot
can retrieve from them." Without running this, your vector_store.py and
retrievers.py have nothing to search -- every query will return empty results.

Usage:
    python scripts/build_indexes.py

Expects this folder layout (create it if it doesn't exist):
    data/raw/code_docs/*.md or *.txt
    data/raw/finance/*.md or *.txt
    data/raw/general_kb/*.md or *.txt
"""

import sys
from pathlib import Path

# Allows running this script directly (python scripts/build_indexes.py)
# without needing to install the project as a package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.chunker import chunk_text
from app.retrieval.embeddings import embed_texts
from app.retrieval.vector_store import upsert_documents, STATIC_COLLECTIONS

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_domain_files(domain: str) -> list[tuple[str, str]]:
    """
    Returns a list of (filename, raw_text) pairs for every supported file
    under data/raw/<domain>/.
    """
    domain_dir = DATA_ROOT / domain
    if not domain_dir.exists():
        print(f"  [skip] {domain_dir} does not exist yet -- create it and add files.")
        return []

    files = [
        f for f in domain_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return [(f.name, f.read_text(encoding="utf-8", errors="ignore")) for f in files]


def build_domain_index(domain: str) -> None:
    print(f"\nBuilding index for domain: {domain}")

    docs = load_domain_files(domain)
    if not docs:
        print(f"  No files found for '{domain}'. Skipping.")
        return

    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    for filename, raw_text in docs:
        chunks = chunk_text(raw_text)
        all_chunks.extend(chunks)
        # Storing source + chunk_index in metadata means later, when this
        # chunk gets retrieved and shown to the user, you can cite exactly
        # which file (and roughly where in it) the answer came from.
        all_metadata.extend(
            {"source": filename, "chunk_index": i} for i in range(len(chunks))
        )
        print(f"  {filename}: {len(chunks)} chunks")

    if not all_chunks:
        print(f"  All files in '{domain}' were empty. Skipping.")
        return

    print(f"  Embedding {len(all_chunks)} total chunks...")
    vectors = embed_texts(all_chunks)

    print(f"  Writing to Qdrant collection '{domain}'...")
    upsert_documents(
        collection_name=domain,
        chunks=all_chunks,
        vectors=vectors,
        metadata=all_metadata,
    )
    print(f"  Done: {len(all_chunks)} chunks indexed into '{domain}'.")


def main():
    print("Building static domain indexes from data/raw/ ...")
    for domain in STATIC_COLLECTIONS:
        build_domain_index(domain)
    print("\nAll done. You can now run retrieval queries against these domains.")


if __name__ == "__main__":
    main()