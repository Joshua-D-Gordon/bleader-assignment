"""Group parser Blocks into retrieval Chunks.

Strategy: one chunk per section. A section is a heading plus all body blocks under
it, up to the next heading of equal-or-shallower level. Tables are always emitted
as their own standalone chunk (never split, never merged into prose) so the
markdown grid survives intact for citation and display. Prose sections larger than
`MAX_TOKENS` are sub-split with LangChain's RecursiveCharacterTextSplitter.

Each chunk keeps `raw_text` pristine (this is what gets cited/shown); the
embedding input is assembled later in the contextual-retrieval step.
"""
from __future__ import annotations

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Block, Chunk

MAX_TOKENS = 600
_OVERLAP_TOKENS = 80
# Text chunks below this carry no retrievable content (bare title fragments,
# stray labels). Tables are exempt — even a small table is meaningful.
MIN_TEXT_TOKENS = 5

_enc = tiktoken.get_encoding("cl100k_base")


def _ntokens(text: str) -> int:
    return len(_enc.encode(text))


def _splitter() -> RecursiveCharacterTextSplitter:
    # Token-aware splitting so MAX_TOKENS is a real bound, not a char guess.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=MAX_TOKENS,
        chunk_overlap=_OVERLAP_TOKENS,
    )


def _emit_text_chunk(chunks: list[Chunk], doc_id: str, section_path: list[str],
                     location: str, order_index: int, text: str) -> None:
    """Emit one or more text chunks, sub-splitting if over the token budget."""
    text = text.strip()
    if _ntokens(text) < MIN_TEXT_TOKENS:
        return
    if _ntokens(text) <= MAX_TOKENS:
        parts = [text]
    else:
        parts = _splitter().split_text(text)
    for j, part in enumerate(parts):
        suffix = f"#{j}" if len(parts) > 1 else ""
        chunks.append(Chunk(
            chunk_id=f"{doc_id}-{order_index}{suffix}",
            doc_id=doc_id,
            chunk_type="text",
            section_path=section_path,
            location=location,
            raw_text=part,
            order_index=order_index,
        ))


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not blocks:
        return chunks

    doc_id = blocks[0].doc_id

    # Accumulator for the current prose section.
    cur_heading: Block | None = None
    cur_body: list[Block] = []
    cur_start_index = 0

    def flush() -> None:
        nonlocal cur_body
        pieces: list[str] = []
        section_path: list[str] = []
        location = ""
        if cur_heading is not None:
            pieces.append(cur_heading.text)
            section_path = cur_heading.section_path
            location = cur_heading.location
        for b in cur_body:
            pieces.append(b.text)
        if not section_path and cur_body:
            section_path = cur_body[0].section_path
            location = cur_body[0].location
        text = "\n\n".join(p for p in pieces if p.strip())
        _emit_text_chunk(chunks, doc_id, section_path, location,
                         cur_start_index, text)
        cur_body = []

    for b in blocks:
        if b.block_type == "table":
            # Tables stand alone. Flush any pending prose first to preserve order.
            flush()
            cur_heading = None
            chunks.append(Chunk(
                chunk_id=f"{doc_id}-{b.order_index}",
                doc_id=doc_id,
                chunk_type="table",
                section_path=b.section_path,
                location=b.location,
                raw_text=b.text,
                order_index=b.order_index,
            ))
            continue

        if b.block_type == "heading":
            # New section boundary: flush the previous one.
            flush()
            cur_heading = b
            cur_start_index = b.order_index
            continue

        # paragraph
        if cur_heading is None and not cur_body:
            cur_start_index = b.order_index
        cur_body.append(b)

    flush()
    return chunks


if __name__ == "__main__":  # standalone smoke test
    import sys
    from .parsers.pdf_parser import parse_pdf
    from .parsers.docx_parser import parse_docx

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for doc_id, parser, path in [
        ("A", parse_pdf, "samples/FDS_PriceBook_Automation_V0.pdf"),
        ("B", parse_docx, "samples/FDS_PriceBook_Automation_V5.docx"),
    ]:
        if which not in ("both", doc_id):
            continue
        cs = chunk_blocks(parser(path, doc_id))
        toks = [_ntokens(c.raw_text) for c in cs]
        tables = [c for c in cs if c.chunk_type == "table"]
        print(f"Doc {doc_id}: {len(cs)} chunks ({len(tables)} tables) "
              f"tokens min/avg/max = {min(toks)}/{sum(toks)//len(toks)}/{max(toks)}")
        over = [t for t in toks if t > MAX_TOKENS]
        print(f"  chunks over {MAX_TOKENS} tokens: {len(over)}")
