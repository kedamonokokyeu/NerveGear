"""
rag/chunk.py

Splits documents into retrievable chunks for RAG.

Targets a character budget but prefers to cut on paragraph and sentence
boundaries. Adds a small overlap between consecutive chunks so facts that straddle
a boundary still appear whole in at least one chunk.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str        # which document it came from
    index: int         # ordinal within that document (0,1,2,...)


def _split_sentences(text: str) -> list[str]:
    # lightweight sentence splitter: break after . ! ? followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _hard_wrap(unit: str, size: int) -> list[str]:
    """
    Prevent chunk sizes from blowing up. 
    Build words that can't be broken via _split_sentences up until size limit, then
    push the current buffer and continue.
    """
    if len(unit) <= size:
        return [unit]
    out, cur = [], ""
    for w in unit.split():
        if cur and len(cur) + len(w) + 1 > size:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(
    text: str,
    source: str = "doc",
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Greedy paragraph/sentence packing into ~chunk_size-character chunks with
    `overlap` characters of trailing context carried into the next chunk."""
    # first step: chunk up the paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            units.append(p)
        else:
            for s in _split_sentences(p):
                units.extend(_hard_wrap(s, chunk_size))

    chunks: list[Chunk] = []
    buf = ""
    # chunk up sentences (mainly if they exceed chunk size)
    for unit in units:
        if buf and len(buf) + len(unit) + 1 > chunk_size:
            chunks.append(buf.strip())
            # carry the tail of the current buffer as overlap into the next chunk
            buf = (buf[-overlap:] + " " + unit) if overlap else unit
        else:
            buf = f"{buf} {unit}".strip() if buf else unit
    if buf.strip():
        chunks.append(buf.strip())

    return [Chunk(text=c, source=source, index=i) for i, c in enumerate(chunks)]


def chunk_markdown(
    text: str,
    source: str = "doc",
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Structure-aware chunking for markdown: split on headings first so a chunk
    never straddles two sections, and prefix each chunk with its heading path
    ("Manufacturing rules > DRIE aspect ratio: ...") so the retriever and the
    reader both see where a passage lives."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush():
        if buf and any(line.strip() for line in buf):
            crumb = " > ".join(t for _, t in path)
            sections.append((crumb, buf.copy()))
        buf.clear()

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            lvl = len(m.group(1))
            path = [(l, t) for l, t in path if l < lvl] + [(lvl, m.group(2).strip())]
        else:
            buf.append(line)
    flush()

    out: list[Chunk] = []
    for crumb, body_lines in sections:
        body = "\n".join(body_lines).strip()
        prefix = f"{crumb}: " if crumb else ""
        for c in chunk_text(body, source=source, chunk_size=max(200, chunk_size - len(prefix)),
                            overlap=overlap):
            out.append(Chunk(text=prefix + c.text, source=source, index=len(out)))
    if not out:                                     # no headings at all
        return chunk_text(text, source=source, chunk_size=chunk_size, overlap=overlap)
    return out
