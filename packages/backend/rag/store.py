"""
rag/store.py

Stores chunk embeddings and searches them by nearest-neighbor lookup.
Since all vectors are L2-normalized, cosine similarity is just a dot product.

Chunks:
[
  Chunk(text="paragraph about fluid dynamics", source="docA", index=0),
  Chunk(text="section about heat transfer", source="docA", index=1),
  ...
]
Embedding + Text Chunk it belongs to for retrieval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np

from .chunk import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float       # cosine similarity in [-1, 1]; higher = more relevant


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = np.zeros((0, dim), dtype=np.float64)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """
        Embeddings is the matrix containing the vectors for each text document.
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings and chunks length mismatch")
        if embeddings.shape[0] == 0:
            return
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {embeddings.shape[1]}")
        self._vecs = np.vstack([self._vecs, embeddings])
        self._chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[Hit]:
        """Provide query embedding --> Return the top-k chunks most similar to the query vector."""
        if len(self) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float64).reshape(-1) # flatten into 1 vector
        sims = self._vecs @ q       # dot product to find relation, which chunks points most similar direction to query vector
        k = min(k, len(self))
        # argpartition for top-k, then sort those k by score descending
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])] # sort by similarity
        return [Hit(chunk=self._chunks[i], score=float(sims[i])) for i in idx] # chunk + score

    def search_mmr(self, query_vec: np.ndarray, k: int = 4, fetch: int = 20, lambda_: float = 0.6) -> list[Hit]:
        """
        Fetch nearest, then compute gram matrix (matrix * transposed matrix) called gram.
        gram[i][j] = similarity between candidate i and j, if too similar, ignore. Most diverse ones chosen.
        """
        if len(self) == 0:
            return []
        from .rerank import mmr_select

        q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
        sims = self._vecs @ q
        fetch = min(fetch, len(self))
        cand = np.argpartition(-sims, fetch - 1)[:fetch]
        cand = cand[np.argsort(-sims[cand])]
        sub = self._vecs[cand]                 # (fetch, d)
        gram = sub @ sub.T                     # candidate-candidate similarity
        picked = mmr_select(sims[cand], gram, k=k, lambda_=lambda_)
        return [Hit(chunk=self._chunks[cand[i]], score=float(sims[cand[i]])) for i in picked]

    def save(self, path: str) -> None:
        """Save vector to disk, no need to recompute embeddings."""
        np.savez(path + ".npz", vecs=self._vecs)
        meta = [{"text": c.text, "source": c.source, "index": c.index} for c in self._chunks]
        with open(path + ".json", "w") as f:
            json.dump({"dim": self.dim, "chunks": meta}, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Re-open vector store and instantly search."""
        with open(path + ".json") as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        store._vecs = np.load(path + ".npz")["vecs"]
        store._chunks = [Chunk(**c) for c in meta["chunks"]]
        return store
