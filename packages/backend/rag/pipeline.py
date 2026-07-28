"""
rag/pipeline.py

Ties chunking, embeddings, and the vector store into a full RAG pipeline.

Index documents once, then answer questions grounded in those documents with
citations. Knowledge updates by re-indexing, not re-training.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Index, rrf_fuse # best match for wide-cast net catch before CrossEncoder, and fuses rankings
from .chunk import Chunk, chunk_markdown, chunk_text # prevent overly large text chunks---just enough context, but not too much
from .store import Hit, VectorStore # matrix embeddings overall for text documents + load/save


@dataclass
class Document:
    id: str
    text: str
    source: str = ""


class RagPipeline:
    def __init__(self, embedder, store: VectorStore | None = None, reranker=None):
        self.embedder = embedder
        self.store = store or VectorStore(dim=embedder.dim)
        self.bm25 = BM25Index()
        self.chunks: list[Chunk] = []          # aligned with store + bm25 ids
        self.reranker = reranker               # optional cross-encoder (rag/rerank.py)

    def index_documents(self, docs: list[Document], chunk_size: int = 800, overlap: int = 150, markdown: bool | None = None) -> int:
        """
        Chunk, embed, and add every document (dense store + BM25 sidecar).
        `markdown=True` uses heading-aware chunking; None auto-detects per doc.
        """
        all_chunks: list[Chunk] = []
        for d in docs:
            src = d.source or d.id
            use_md = markdown if markdown is not None else ("\n#" in ("\n" + d.text))
            fn = chunk_markdown if use_md else chunk_text # picks chunking function
            all_chunks.extend(fn(d.text, source=src, chunk_size=chunk_size, overlap=overlap)) # all documents transformed into chunks
        if not all_chunks:
            return 0
        vecs = self.embedder.embed([c.text for c in all_chunks]) # embed --> form vector assignments
        self.store.add(vecs, all_chunks)
        self.bm25.add([c.text for c in all_chunks]) # only tokenizes --- ready for further processing
        self.chunks.extend(all_chunks) # flat list for double check
        return len(all_chunks)

    def retrieve_hybrid(self, question: str, k: int = 4, fetch: int = 24) -> list[Hit]:
        """
        Dense + BM25, fused by reciprocal rank (RRF), optionally reordered by
        a cross-encoder. The build plan's retrieval stack in one call.
        Hit gives each chunk a score. So it's {Hit value, Score}
        """
        dense = self.retrieve(question, k=min(fetch, max(1, len(self.chunks)))) # embedded vector + search
        dense_ids = [self.chunks.index(h.chunk) if h.chunk in self.chunks else -1
                     for h in dense] # each Hit contains chunk, find index in self.chunks, if not found, use -1
        dense_ids = [i for i in dense_ids if i >= 0] # get rid of invalids
        lex = [i for i, _ in self.bm25.search(question, k=fetch)] # lexical search
        fused = rrf_fuse([dense_ids, lex])
        cand = [self.chunks[i] for i in fused[:max(k * 3, k)]] # candidate chunks
        if self.reranker is not None and cand:
            order = self.reranker.rank(question, [c.text for c in cand])
            cand = [cand[i] for i in order]
        dense_scores = {id(h.chunk): h.score for h in dense}
        return [Hit(chunk=c, score=float(dense_scores.get(id(c), 0.0)))
                for c in cand[:k]] # return best chunks and scores

    def retrieve(self, question: str, k: int = 4) -> list[Hit]:
        qv = self.embedder.embed([question])[0] # nested list, grab the actual embedding vector
        return self.store.search(qv, k=k)

    def retrieve_mmr(self, question: str, k: int = 4, fetch: int = 20, lambda_: float = 0.6) -> list[Hit]:
        """Diversity-aware retrieval (see rag/rerank.py): avoids returning k
        near-duplicate passages."""
        qv = self.embedder.embed([question])[0]
        return self.store.search_mmr(qv, k=k, fetch=fetch, lambda_=lambda_)

    def build_prompt(self, question: str, hits: list[Hit]) -> str:
        """Assemble the context block + citations + question into one prompt. The
        numbered sources let the model (and the reader) cite [1], [2], ..."""
        blocks = []
        for i, h in enumerate(hits, start=1):
            blocks.append(f"[{i}] (source: {h.chunk.source}, score {h.score:.2f})\n{h.chunk.text}")
        context = "\n\n".join(blocks) if blocks else "(no relevant context found)" # one long context string of the best, most relevant chunks retrieved
        return (
            "You are NerveGear's engineering assistant. Answer the question using ONLY "
            "the context below. Cite sources inline as [1], [2]. If the context does "
            "not contain the answer, say so plainly.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )

    def answer(self, question: str, k: int = 4, generator=None, mode: str = "hybrid") -> dict:
        """Full query. `generator` is an optional callable(prompt) -> str (wire your
        LLM here). Returns the answer (or the assembled prompt if no generator),
        plus the citations so the UI can show sources."""
        hits = (self.retrieve_hybrid(question, k=k) if mode == "hybrid"
                else self.retrieve(question, k=k))
        prompt = self.build_prompt(question, hits)
        if generator is not None:
            text = generator(prompt)
        else: # return prompt if no llm provided
            text = (
                "[no generator wired] Retrieved context is below; pass a "
                "generator=callable(prompt)->str (e.g. a Claude API call) to get a " 
                "written answer.\n\n" + prompt
            )
        return {
            "answer": text,
            "citations": [
                {"n": i + 1, "source": h.chunk.source, "score": round(h.score, 3),
                 "preview": h.chunk.text[:160]}
                for i, h in enumerate(hits)
            ],
        }
