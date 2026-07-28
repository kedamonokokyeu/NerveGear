"""
rag
===
A small, readable Retrieval-Augmented Generation pipeline, built to be learned
from. Five stages, one module each:

    chunk.py    split documents into passages
    embed.py    text -> vectors (offline hashing fallback + optional real model)
    store.py    hold vectors + nearest-neighbour search
    pipeline.py retrieve top-k, assemble a cited prompt, (optional) generate
    corpus.py   load source material (the NerveGear roadmap, a CFD glossary)

See rag/README.md for the full walkthrough.
"""

from .chunk import Chunk, chunk_text
from .embed import HashingEmbedder, SentenceTransformerEmbedder, get_default_embedder
from .pipeline import Document, RagPipeline
from .store import Hit, VectorStore

__all__ = [
    "Chunk", "chunk_text",
    "HashingEmbedder", "SentenceTransformerEmbedder", "get_default_embedder",
    "Document", "RagPipeline",
    "Hit", "VectorStore",
]
