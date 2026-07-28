"""
rag/embed.py

Turns text into vectors so similar meanings sit close together in embedding space.

There are two embedders. HashingEmbedder (for testing mainly). SentenceTransformerEmbeder gives semantic embeddings for production.
SentenceTransformer already has a neural network that encodes semantic meaning.
"""

from __future__ import annotations
import hashlib
import re
import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

# semantic embedder doesn't need useless stopwords
# lexical hashing fallback (HashingEmbedder) it does help though
_STOP = frozenset(
    "a an and the of to in on at for is are was were be been being this that these "
    "those it its as by with from or not no but if then than so we you they i he she "
    "do does did has have had will would can could should may might into out over "
    "under about above below up down off can't what which who whom whose how when "
    "where why".split()
)


def _tokenize(text: str) -> list[str]:
    """Returns clean tokens for embeddings."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _l2(mat: np.ndarray) -> np.ndarray:
    """
    One vector per text chunk. In embed() of HashingEmbedder, the vectors
    are vertically stacked (using np.vstack), and this is the mat parameter (short for matrix)
    being fed in.
    The dimensions of the chunks are encoded by the model for semantic meaning. 
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True) # computes euclidean length of each row vector
    norms[norms == 0] = 1.0 # if vector, e.g. (0 0 0) then can't divide by 0, so set to 1
    return mat / norms # normalize it


class HashingEmbedder:
    """
    Hashing-trick bag-of-words embedder. Deterministic and offline.
    Each vector of 512 dimensions is for the entire text/paragraph.
    Token signals stored in buckets.
    """

    def __init__(self, dim: int = 512): # limit to 512 dimensions
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 # maps hash into a number between 0 and dim - 1
            # a signed second hash reduces collisions cancelling signal
            v[(h // self.dim) % self.dim] += 0.5
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        """Stack all the text embeddings, then send it to be normalized. Comparing each text/paragraph embedding vector."""
        mat = np.vstack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim))
        return _l2(mat)


class SentenceTransformerEmbedder:
    """Real semantic embeddings. Requires `pip install sentence-transformers`."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim)) # 0 rows, 512 columns, actually empty but ready to hold up to that much data
        # matrix of shape(len(texts), dim) --- one row per text
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float64)


def get_default_embedder(dim: int = 512):
    """Prefer real semantic embeddings; fall back to the offline hashing one."""
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashingEmbedder(dim=dim)
