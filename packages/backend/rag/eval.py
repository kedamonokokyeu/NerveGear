"""
rag/eval.py

Measures retrieval quality with hit@k.

Given labeled question → expected-source pairs, checks what fraction of the time
the right document appears in the top-k retrieved chunks. Run whenever you change
the embedder, chunk size, or reranker to catch regressions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    hit_at_k: float                 # fraction of queries whose expected source was retrieved
    k: int
    n: int
    misses: list                    # questions that missed, for inspection


def hit_at_k(pipeline, labeled: list[tuple[str, str]], k: int = 4, use_mmr: bool = False) -> EvalResult:
    """`labeled` = [(question, expected_source_substring), ...]. A query 'hits' if
    any of the top-k retrieved chunks' source contains the expected substring."""
    hits = 0
    misses = []
    for question, expected in labeled:
        if use_mmr:
            qv = pipeline.embedder.embed([question])[0]
            results = pipeline.store.search_mmr(qv, k=k)
        else:
            results = pipeline.retrieve(question, k=k)
        if any(expected.lower() in h.chunk.source.lower() for h in results):
            hits += 1
        else:
            misses.append(question)
    n = len(labeled)
    return EvalResult(hit_at_k=hits / n if n else 0.0, k=k, n=n, misses=misses)


def run_golden(pipeline=None, path: str | None = None, hybrid: bool = True) -> EvalResult:
    """Evaluate against rag/golden.json. Builds the default corpus pipeline if
    none is given. The CI gate: hit@k must stay ≥ 0.8."""
    import json
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")
    with open(path, encoding="utf-8") as f:
        gold = json.load(f)
    if pipeline is None:
        from .corpus import nervegear_corpus
        from .embed import get_default_embedder
        from .pipeline import RagPipeline
        pipeline = RagPipeline(get_default_embedder())
        pipeline.index_documents(nervegear_corpus())
    labeled = [tuple(c) for c in gold["cases"]]
    k = int(gold.get("k", 4))
    if hybrid:
        hits = 0
        misses = []
        for q, expected in labeled:
            res = pipeline.retrieve_hybrid(q, k=k)
            if any(expected.lower() in h.chunk.source.lower() for h in res):
                hits += 1
            else:
                misses.append(q)
        return EvalResult(hit_at_k=hits / len(labeled), k=k, n=len(labeled), misses=misses)
    return hit_at_k(pipeline, labeled, k=k)


if __name__ == "__main__":
    r = run_golden()
    print(f"golden hit@{r.k}: {r.hit_at_k:.2f} ({r.n} cases)")
    for m in r.misses:
        print("  MISS:", m)
