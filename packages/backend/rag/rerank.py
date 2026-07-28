"""
rag/rerank.py

Reranks retrieved chunks after the nearest-neighbor fetch to reduce near-duplicates.

Implements MMR (Maximal Marginal Relevance): instead of returning the k most
similar chunks (which are often near-duplicates), it balances relevance to the
query against diversity from what's already been selected.
"""

from __future__ import annotations

import numpy as np


def mmr_select(sim_to_query: np.ndarray, gram: np.ndarray, k: int, lambda_: float = 0.6) -> list[int]:
    """
    `sim_to_query[i]` = similarity of candidate i to the query;
    `gram[i, j]` = similarity between candidates i and j. Returns indices of the
    chosen candidates, best first.
    """
    n = len(sim_to_query)
    k = min(k, n)
    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda r: sim_to_query[r])
        else:
            def score(r):
                diversity_penalty = max(gram[r, s] for s in selected)
                return lambda_ * sim_to_query[r] - (1 - lambda_) * diversity_penalty
            best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return selected


class CrossEncoderReranker:
    """
    Second reranker. CrossEncoder reads both query and document at the same time.
    By comparing token to token, it can identify what sounds contradictory.
        rr = CrossEncoderReranker.try_create()   # None if package missing
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy heavy import
        self.model = CrossEncoder(model_name)

    @classmethod
    def try_create(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            return cls(model_name)
        except Exception:  # noqa: BLE001 - package or weights unavailable
            return None

    def rank(self, question: str, passages: list[str]) -> list[int]:
        """
        Build pairs: (question, passage)
        Predict scores using cross encoder --> sort passage indices by descending score --> return ranking
        """
        scores = self.model.predict([(question, p) for p in passages])
        return sorted(range(len(passages)), key=lambda i: -float(scores[i]))
