"""
rag/bm25.py

SentenceTransformer captures semantic meaning. Purpose of this file
is to capture exact keyword matches. Ranks documentsby how well they match a keyword query.
Scores based on how relevant document d is to query q based on token matches. Rare terms also carry more weight. 
Penalizes long documents, rewards rare, repeated terms more, etc.
Built for hybrid retrieval, using both.
"""

from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    # keep short tokens: "Re", "Nu", "dP" are exactly the terms that matter here
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1 
        self.b = b
        self.docs: list[list[str]] = []
        self.df: dict[str, int] = {}
        self.avgdl = 0.0 # avg doc length

    def add(self, texts: list[str]) -> None:
        for t in texts:
            toks = tokenize(t)
            self.docs.append(toks)
            for w in set(toks):
                self.df[w] = self.df.get(w, 0) + 1
        self.avgdl = (sum(len(d) for d in self.docs) / len(self.docs)) if self.docs else 0.0

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """
        [(doc_index, score), ...] best first. Scores are query-dependent;
        use ranks (e.g. RRF), not raw values, when fusing with other systems.
        """
        q = tokenize(query)
        n = len(self.docs)
        if not n or not q:
            return []
        scores = [0.0] * n
        for w in q:
            df = self.df.get(w)
            if not df:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5)) # compute IDF part of formula (reference doc)
            # score documents --- order them
            for i, doc in enumerate(self.docs):
                tf = doc.count(w)
                if not tf:
                    continue
                dl = len(doc)
                scores[i] += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
        order = sorted(range(n), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in order[:k] if scores[i] > 0.0]


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[int]:
    """
    Reciprocal-rank fusion: score(d) = Σ 1/(k + rank). 
    It's rank-based, so other score scales will fuse. Reference doc for more info.
    """
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda d: -score[d])
