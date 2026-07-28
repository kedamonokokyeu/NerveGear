"""RAG pipeline: chunking, embeddings, vector store, end-to-end retrieval.

Uses the dependency-free HashingEmbedder so these run anywhere."""

from __future__ import annotations

import numpy as np

from rag.chunk import chunk_text
from rag.corpus import cfd_glossary
from rag.embed import HashingEmbedder
from rag.pipeline import RagPipeline
from rag.store import VectorStore


# ---- chunking -------------------------------------------------------------
def test_chunking_respects_size_and_covers_text():
    text = ("Paragraph one is short.\n\n" + ("word " * 400) + "\n\nFinal short paragraph.")
    chunks = chunk_text(text, source="t", chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    assert all(c.source == "t" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # no chunk wildly exceeds the budget (allow overlap slack)
    assert max(len(c.text) for c in chunks) <= 300 + 200


def test_chunking_overlap_carries_context():
    text = " ".join(f"s{i}." for i in range(200))
    no = chunk_text(text, chunk_size=200, overlap=0)
    yes = chunk_text(text, chunk_size=200, overlap=80)
    # overlap produces at least as many chunks (repeats tail context)
    assert len(yes) >= len(no)


# ---- embeddings -----------------------------------------------------------
def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(dim=256)
    a = e.embed(["reynolds number measures inertial to viscous forces"])
    b = e.embed(["reynolds number measures inertial to viscous forces"])
    assert a.shape == (1, 256)
    assert np.allclose(a, b)                       # deterministic
    assert np.isclose(np.linalg.norm(a[0]), 1.0)   # L2-normalized


def test_similar_text_scores_higher_than_unrelated():
    e = HashingEmbedder(dim=512)
    v = e.embed([
        "the reynolds number is the ratio of inertial to viscous forces",
        "reynolds number inertial viscous forces ratio fluid",
        "a recipe for chocolate chip cookies with butter and sugar",
    ])
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


# ---- store ----------------------------------------------------------------
def test_store_search_returns_most_similar():
    e = HashingEmbedder(dim=512)
    store = VectorStore(dim=512)
    docs = ["alpha beta gamma", "delta epsilon zeta", "alpha beta delta"]
    from rag.chunk import Chunk
    chunks = [Chunk(text=t, source="s", index=i) for i, t in enumerate(docs)]
    store.add(e.embed(docs), chunks)
    hits = store.search(e.embed(["alpha beta"])[0], k=2)
    assert len(hits) == 2
    assert "alpha beta" in hits[0].chunk.text          # closest match first
    assert hits[0].score >= hits[1].score


def test_store_save_load_roundtrip(tmp_path):
    e = HashingEmbedder(dim=128)
    store = VectorStore(dim=128)
    from rag.chunk import Chunk
    chunks = [Chunk(text="hello world", source="s", index=0)]
    store.add(e.embed(["hello world"]), chunks)
    p = str(tmp_path / "idx")
    store.save(p)
    loaded = VectorStore.load(p)
    assert len(loaded) == 1
    assert loaded._chunks[0].text == "hello world"


# ---- end to end -----------------------------------------------------------
def test_pipeline_retrieves_relevant_glossary_entry():
    pipe = RagPipeline(HashingEmbedder(dim=512))
    n = pipe.index_documents(cfd_glossary())
    assert n >= 6
    hits = pipe.retrieve("what does the reynolds number mean?", k=3)
    assert hits
    # the reynolds entry should be among the top-3 retrieved
    assert any("reynolds" in h.chunk.source.lower() for h in hits)


def test_answer_without_generator_returns_context_and_citations():
    pipe = RagPipeline(HashingEmbedder(dim=512))
    pipe.index_documents(cfd_glossary())
    out = pipe.answer("explain inlet velocity", k=2)
    assert "citations" in out and len(out["citations"]) == 2
    assert out["citations"][0]["n"] == 1
    assert "QUESTION:" in out["answer"]     # assembled prompt present


def test_answer_with_generator_invokes_it():
    pipe = RagPipeline(HashingEmbedder(dim=512))
    pipe.index_documents(cfd_glossary())
    out = pipe.answer("what is a surrogate model?", k=2, generator=lambda p: "STUB ANSWER")
    assert out["answer"] == "STUB ANSWER"
