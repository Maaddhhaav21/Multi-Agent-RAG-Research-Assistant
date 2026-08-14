"""
chunker.py
----------
Splits raw text files into smaller overlapping pieces ("chunks") before embedding.

Why chunk at all? Two reasons:
1. Embedding models have a max input length -- a whole document often won't fit.
2. Retrieval precision -- if you embed a whole 5000-word doc as ONE vector, a
   query about one paragraph in it gets diluted by everything else in that
   vector. Smaller chunks mean more precise "this is the relevant bit" matches.

Why OVERLAPPING chunks (not just cutting every N words)? If a sentence that
answers the user's question happens to fall right at a chunk boundary, a
non-overlapping split can cut it in half and lose the meaning. A small overlap
(e.g. 50 words) means most sentences appear whole in at least one chunk.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[str]:
    """
    Simple word-count-based chunker. Not as smart as a sentence-aware or
    semantic chunker, but it's easy to understand and good enough to get
    your pipeline working end-to-end -- you can swap in something fancier
    (e.g. langchain's RecursiveCharacterTextSplitter) later without changing
    anything else in the pipeline.

    chunk_size: target number of words per chunk
    overlap: number of words repeated between consecutive chunks
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break
        # Move forward by (chunk_size - overlap), not chunk_size, so the
        # next chunk starts BEFORE the previous one ends -- that's the overlap.
        start += chunk_size - overlap

    return chunks