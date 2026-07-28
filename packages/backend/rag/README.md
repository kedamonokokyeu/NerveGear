# NerveGear RAG — the engineering assistant's retrieval pipeline

## 1. What it does, in one picture

```
  DOCUMENTS (knowledge base, repo docs, roadmap)
        │
        │  chunk_markdown / chunk_text          (chunk.py)
        ▼
     CHUNKS ──embed──► dense vectors ──► VectorStore   (embed.py, store.py)
        │
        └──tokenize──► BM25 index                       (bm25.py)
                                                  ── index time (once) ──

  QUESTION
     │
     ├─ dense search (cosine)      ─┐
     │                              ├─ RRF fuse ─► cross-encoder rerank ─► top-k
     └─ BM25 search (keywords)     ─┘   (rrf_fuse)     (optional)         chunks
                                                                            │
                              build_prompt (numbered, cited context) ◄──────┘
                                                                            │
                              generator (Claude)  ─OR─  extractive fallback │
                                                                            ▼
                                             ANSWER  +  CITATIONS
                                                  ── query time ──
```

Two ideas make it good:
- **Hybrid retrieval** — semantic search (understands paraphrase) *and* keyword
  search (nails exact terms like `Re`, `Nu`, `dP`, `DRIE`), fused together. Either
  alone misses cases the other catches.
- **Grounding + citations** — the model answers *only* from retrieved context and
  cites `[1]`, `[2]`, so answers are checkable and update when docs change.

---

## 2. The retrieval flow, stage by stage

### ① Chunking — `chunk.py`
Documents are split into passages small enough to embed meaningfully but large
enough to carry context.

- **`chunk_text(text, chunk_size=800, overlap=150)`** — greedy packing on
  paragraph → sentence boundaries, with `overlap` characters of trailing context
  carried into the next chunk so a fact straddling a boundary survives whole.
  `_hard_wrap` guarantees no chunk blows past the size budget.
- **`chunk_markdown(text, chunk_size=900, overlap=120)`** — **structure-aware.**
  It splits on markdown headings first (so a chunk never straddles two sections)
  and **prefixes each chunk with its heading path** — e.g.
  `"Manufacturing rules > DRIE aspect ratio: ..."`. That breadcrumb helps both the
  retriever *and* the reader see where a passage lives. Falls back to `chunk_text`
  if the document has no headings.
- `index_documents` **auto-detects** markdown (looks for a `\n#` heading) per
  document, so mixed corpora just work.

Each chunk is a `Chunk(text, source, index)`.

### ② Embedding — `embed.py`
Turns text into vectors so similar meanings sit close together. All vectors are
**L2-normalized**, which makes cosine similarity a plain dot product downstream.

- **`SentenceTransformerEmbedder`** (default when available) — real semantic
  embeddings via `all-MiniLM-L6-v2` (384-dim). Understands paraphrase/synonymy.
- **`HashingEmbedder`** (offline fallback) — deterministic bag-of-words hashed
  into 512 dims, with stopword filtering. No downloads, no network; used for tests
  and any environment without `sentence-transformers`. It captures shared *words*,
  not deep meaning — which is exactly why BM25 (below) complements it.
- **`get_default_embedder()`** — returns the real one if installed, else hashing.
  The `VectorStore` is sized to `embedder.dim`, so both just work.

### ③ Dense store — `store.py`
- **`VectorStore`** holds the embedding matrix + the chunks. Because vectors are
  normalized, `search(query_vec, k)` is one matrix–vector product (`vecs @ q`) —
  exact and instant for this corpus size, no ANN index needed.
- **`search_mmr(...)`** — diversity-aware variant (see MMR below).
- **`save(path)` / `load(path)`** — persist the index (`.npz` vectors + `.json`
  metadata) so you don't re-embed on every start.
- Returns `Hit(chunk, score)` where score is cosine similarity in `[-1, 1]`.

### ④ Lexical store — `bm25.py`
Keyword retrieval that catches exact terms embeddings can miss.

- **`BM25Index(k1=1.5, b=0.75)`** — Okapi BM25: rewards rare, repeated query terms
  (IDF), saturates term frequency (`k1`), and normalizes for document length (`b`).
  The tokenizer **keeps short tokens** (`re`, `nu`, `dp`) because those are the
  domain's most important symbols.
- **`rrf_fuse(rankings, k=60)`** — **Reciprocal Rank Fusion.** Combines the dense
  and BM25 result *rankings* (not their raw scores, which live on different scales)
  via `score(d) = Σ 1/(k + rank)`. This is how the two retrievers become one list.

### ⑤ Reranking — `rerank.py`
Reorders candidates after the cheap fetch.

- **`mmr_select(...)`** — Maximal Marginal Relevance: balances relevance to the
  query against **diversity** from already-picked chunks (`lambda_=0.6`), so you
  don't return k near-duplicate passages. Powers `store.search_mmr` /
  `pipeline.retrieve_mmr`.
- **`CrossEncoderReranker`** (optional) — a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads the *query and passage together*
  and scores true relevance far better than embeddings. Heavy, so it's opt-in;
  `CrossEncoderReranker.try_create()` returns `None` if the package/weights aren't
  available, and the pipeline simply skips it.

### ⑥ Orchestration — `pipeline.py`
`RagPipeline(embedder, store=None, reranker=None)` ties it together.

- **`index_documents(docs, chunk_size=800, overlap=150, markdown=None)`** — chunk
  every doc, embed → dense store, tokenize → BM25 sidecar, keep a flat aligned
  chunk list. Returns the number of chunks.
- **`retrieve(question, k=4)`** — dense only.
- **`retrieve_hybrid(question, k=4, fetch=24)`** — the production path: dense +
  BM25, fused by RRF, then optionally reordered by the cross-encoder, return top-k.
- **`retrieve_mmr(question, k=4, fetch=20, lambda_=0.6)`** — diversity-aware dense.
- **`build_prompt(question, hits)`** — assembles a numbered, cited context block +
  the instruction *"answer using ONLY this context, cite [1], [2], say so if it's
  not here"* + the question.
- **`answer(question, k=4, generator=None, mode="hybrid")`** — retrieve → prompt →
  generate. Returns `{"answer", "citations": [{n, source, score, preview}, ...]}`.
  With no generator it returns the assembled, cited context (extractive) — still
  useful and dependency-free.

### ⑦ Generation — `generate.py`
Turns the cited context into a written answer. Optional — the SDK is imported
lazily, so importing this module never needs the package or a key.

- **`claude_generator(model="claude-sonnet-4-6", ...)`** — a `generator(prompt)->str`
  backed by the Anthropic SDK.
- **`default_generator(model="claude-haiku-4-5-...")`** — "best available":
  returns `None` if `ANTHROPIC_API_KEY` is unset (pipeline stays extractive), uses
  the SDK if installed, or falls back to a **dependency-free `urllib` client** that
  calls the Anthropic HTTP API directly.

---

## 3. The corpus & knowledge base — `corpus.py`, `knowledge/`

- **`nervegear_corpus()`** — the assistant's full knowledge: the built-in fluid
  glossary **plus** `knowledge/microfluidics.md` **plus** the repo docs
  (`CONTRACT.md`, `README.md`, `PROJECT_MAP.md`, `MICROSCALE_PIVOT.md`,
  `THERMAL.md`, `MODEL_CARD.md`) **plus** the roadmap `.docx` when present. Missing
  files are skipped silently, so it works in any checkout.
- **`cfd_glossary()`** — a small built-in definition set so the pipeline always has
  something to retrieve, even with no files on disk.
- **`load_docx(path)`** — pulls plain text from a `.docx` (needs `python-docx`).
- **`knowledge/microfluidics.md`** — the curated, heading-structured knowledge base
  (physics, DRIE manufacturing rules, and the meaning of every number the UI
  shows). This is the primary thing to grow as the product evolves — add a section
  here and it's immediately retrievable.

---

## 4. Evaluation — `eval.py`, `golden.json`

You can't improve retrieval you don't measure.

- **`golden.json`** — a labeled set of `[question, expected_source_substring]`
  cases (currently 14, `k=4`). A query "hits" if the expected source appears in the
  top-k retrieved chunks.
- **`run_golden(...)`** — builds the default corpus pipeline (if none given) and
  scores **hit@k** in hybrid mode. **CI gate: hit@k must stay ≥ 0.80.**
- **`hit_at_k(pipeline, labeled, k, use_mmr=False)`** — the generic scorer.

Run it:
```bash
python -m rag.eval
# -> golden hit@4: 0.93 (14 cases)   + any MISS: lines to inspect
```
Re-run after any change to the embedder, chunk size, reranker, or knowledge base.

---

## 5. Quickstart

```python
from rag.pipeline import RagPipeline
from rag.embed import get_default_embedder
from rag.corpus import nervegear_corpus
from rag.generate import default_generator   # None unless ANTHROPIC_API_KEY is set

pipe = RagPipeline(get_default_embedder())
pipe.index_documents(nervegear_corpus())          # chunk + embed + BM25, once

out = pipe.answer("What are the DRIE etch rules NerveGear enforces?",
                  generator=default_generator())
print(out["answer"])       # grounded, cites [1], [2] (or the extractive context)
print(out["citations"])    # [{n, source, score, preview}, ...] for the UI
```

Add the optional cross-encoder reranker:
```python
from rag.rerank import CrossEncoderReranker
pipe = RagPipeline(get_default_embedder(), reranker=CrossEncoderReranker.try_create())
```

---

## 6. Configuration & knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| `chunk_size` / `overlap` | `index_documents` | 800 / 150 (md: 900 / 120) | passage size vs context bleed |
| `k` | retrieve/answer | 4 | how many chunks reach the prompt |
| `fetch` | `retrieve_hybrid` | 24 | candidate pool before fusion/rerank |
| `k1` / `b` | `BM25Index` | 1.5 / 0.75 | term saturation / length normalization |
| RRF `k` | `rrf_fuse` | 60 | fusion smoothing |
| `lambda_` | MMR | 0.6 | relevance ↔ diversity |
| embedder | `get_default_embedder` | MiniLM (384) / hashing (512) | semantic vs offline |
| generator model | `default_generator` | Haiku / Sonnet | answer quality vs cost |
| `ANTHROPIC_API_KEY` | env | unset | unset → extractive answers |

---

## 7. Extending it

- **Add knowledge** → drop a section into `knowledge/microfluidics.md` (or a new
  file wired into `nervegear_corpus()`), re-index, add a `golden.json` case for it.
- **Better retrieval** → enable the cross-encoder reranker.
- **Better embeddings** → install `sentence-transformers` (auto-used).
- **Written answers** → set `ANTHROPIC_API_KEY` (+ optionally `pip install
  anthropic`).
- **Persistence** → `store.save(path)` / `VectorStore.load(path)` to skip
  re-embedding on startup.

---

## 8. Dependencies

- **Required:** `numpy` (that's it — the pipeline runs offline with the hashing
  embedder and extractive answers).
- **Optional, auto-detected:** `sentence-transformers` (semantic embeddings +
  cross-encoder), `anthropic` (written answers), `python-docx` (roadmap `.docx`).

---

## 9. Design decisions (the "why")

- **Hybrid over pure-vector** — domain queries hinge on exact symbols (`Re`, `Nu`,
  `DRIE`) that dense embeddings blur; BM25 recovers them, and RRF needs no score
  calibration to fuse the two.
- **Markdown-aware chunking** — engineering docs are heading-structured; keeping a
  chunk inside one section and tagging it with its heading path measurably improves
  retrieval and gives the reader provenance.
- **Everything optional degrades gracefully** — no key → extractive; no ST → hashing;
  no cross-encoder → skip. The pipeline never hard-fails on a missing extra.
- **Golden-set gate** — one small, versioned eval keeps retrieval honest as the
  knowledge base grows.
