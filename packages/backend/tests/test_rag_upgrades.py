"""MMR reranker, MMR retrieval, eval harness, and the Claude generator factory."""

from __future__ import annotations

import numpy as np

from rag.chunk import Chunk
from rag.corpus import cfd_glossary
from rag.embed import HashingEmbedder
from rag.eval import hit_at_k
from rag.generate import claude_generator
from rag.pipeline import RagPipeline
from rag.rerank import mmr_select
from rag.store import VectorStore


# ---- MMR (pure) -----------------------------------------------------------
def test_mmr_picks_diverse_over_duplicate():
    # candidates 0 and 1 are identical (a near-duplicate pair), 2 is different.
    # All similarly relevant to the query.
    sim = np.array([0.9, 0.89, 0.85])
    gram = np.array([
        [1.0, 0.99, 0.10],
        [0.99, 1.0, 0.10],
        [0.10, 0.10, 1.0],
    ])
    picked = mmr_select(sim, gram, k=2, lambda_=0.5)
    # should pick the top one, then the DIVERSE one (2), not its duplicate (1)
    assert picked[0] == 0
    assert picked[1] == 2


def test_mmr_lambda_one_is_plain_topk():
    sim = np.array([0.5, 0.9, 0.7])
    gram = np.eye(3)
    picked = mmr_select(sim, gram, k=3, lambda_=1.0)
    assert picked == [1, 2, 0]      # pure relevance order


# ---- MMR retrieval through the store --------------------------------------
def test_store_search_mmr_returns_k():
    e = HashingEmbedder(dim=256)
    store = VectorStore(dim=256)
    texts = ["alpha beta", "alpha beta gamma", "delta epsilon", "zeta eta theta"]
    chunks = [Chunk(text=t, source=f"s{i}", index=0) for i, t in enumerate(texts)]
    store.add(e.embed(texts), chunks)
    hits = store.search_mmr(e.embed(["alpha beta"])[0], k=2, fetch=4)
    assert len(hits) == 2


# ---- eval harness ---------------------------------------------------------
def test_eval_hit_at_k_on_glossary():
    pipe = RagPipeline(HashingEmbedder(dim=512))
    pipe.index_documents(cfd_glossary())
    labeled = [
        ("what is the reynolds number", "reynolds"),
        ("define inlet velocity", "inlet"),
        ("what are the navier stokes equations", "navier"),
    ]
    res = hit_at_k(pipe, labeled, k=3)
    assert res.n == 3
    assert 0.0 <= res.hit_at_k <= 1.0
    assert res.hit_at_k >= 2 / 3      # a decent embedder gets most of these


# ---- generator factory (no API call) --------------------------------------
def test_claude_generator_returns_callable_without_sdk():
    gen = claude_generator()          # must not import anthropic or need a key
    assert callable(gen)
