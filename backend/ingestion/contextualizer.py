"""Contextual Retrieval (Anthropic's technique, applied here with an OpenAI model).

Before embedding, each chunk is prepended with a short LLM-generated blurb that
situates it within its document — "this chunk is from Doc B's System Architecture
section describing the output-generation layer." Embedding `blurb + raw_text`
(instead of raw_text alone) measurably improves retrieval on ambiguous chunks,
because a chunk that only makes sense in context now carries that context in its
vector.

The blurb is derived by the LLM from the chunk + its breadcrumb; nothing about
price books is hardcoded, so this generalizes to any document pair.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..app import config
from ..app.clients import openai_client
from .models import Chunk

_MAX_WORKERS = 8

_SYSTEM = (
    "You situate a document excerpt within its source document to improve search "
    "retrieval. Given the document name, the section breadcrumb, and the excerpt, "
    "write ONE concise sentence (max 30 words) describing what this excerpt is "
    "about and where it sits. Output only that sentence — no preamble."
)


def _blurb_for(chunk: Chunk, doc_display: str) -> str:
    user = (
        f"Document: {doc_display}\n"
        f"Section: {chunk.breadcrumb() or '(top level)'}\n"
        f"Excerpt:\n{chunk.raw_text[:1500]}"
    )
    resp = openai_client().chat.completions.create(
        model=config.CONTEXT_MODEL,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0,
        max_tokens=80,
    )
    return (resp.choices[0].message.content or "").strip()


def contextualize(chunks: list[Chunk]) -> list[Chunk]:
    """Populate `embedding_text = blurb + breadcrumb + raw_text` for each chunk.

    Runs blurb generation concurrently. On any per-chunk failure we fall back to
    the deterministic breadcrumb prefix so ingestion never hard-fails on a blurb.
    """
    def build(chunk: Chunk) -> Chunk:
        display = config.DOCUMENTS[chunk.doc_id]["display"]
        breadcrumb = chunk.breadcrumb()
        try:
            blurb = _blurb_for(chunk, display)
        except Exception:
            blurb = ""
        header = " ".join(p for p in [blurb, breadcrumb] if p)
        chunk.embedding_text = f"{header}\n\n{chunk.raw_text}" if header else chunk.raw_text
        return chunk

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        return list(pool.map(build, chunks))
