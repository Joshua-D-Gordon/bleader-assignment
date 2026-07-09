"""PDF -> list[Block].

Uses pymupdf4llm to render each page to markdown (page_chunks=True gives per-page
text plus metadata). Markdown gives us `#`-prefixed headings and pipe-delimited
tables for free, so we can reconstruct section hierarchy and keep tables whole —
which raw text extraction (pdftotext/pdfplumber) would not.

We walk the markdown line by line, maintaining a heading stack. Consecutive table
lines (`| ... |`) are accumulated into a single `table` block rather than emitted
one row per block.
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf4llm

from ..models import Block

# pymupdf4llm derives heading level from font size. Real section headings in this
# document are H1-H4 and short; anything deeper/longer is body text mislabelled.
_MAX_HEADING_LEVEL = 4
_MAX_HEADING_LEN = 120


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


# Markdown horizontal rules / separators that pymupdf4llm emits between blocks.
_RULE_RE = re.compile(r"^[-_*]{3,}$")


def _is_noise(text: str) -> bool:
    return bool(_RULE_RE.match(text.strip()))


def parse_pdf(path: str | Path, doc_id: str) -> list[Block]:
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)

    blocks: list[Block] = []
    section_path: list[str] = []
    order_index = 0

    for page in pages:
        page_num = page["metadata"].get("page", "?")
        location = f"Page {page_num}"
        lines = page["text"].split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped or _is_noise(stripped):
                i += 1
                continue

            # Heading: leading run of '#'. pymupdf4llm maps font size to heading
            # level, so body paragraphs leak through as level-5+ "headings" (long
            # text). Treat only short, top-level markup as real headings; demote
            # the rest to paragraphs so section_path stays clean.
            if stripped.startswith("#"):
                hashes = len(stripped) - len(stripped.lstrip("#"))
                title = stripped.lstrip("#").strip()
                if title and hashes <= _MAX_HEADING_LEVEL and len(title) <= _MAX_HEADING_LEN:
                    section_path = section_path[: hashes - 1] + [title]
                    blocks.append(Block(
                        doc_id=doc_id, block_type="heading", heading_level=hashes,
                        text=title, section_path=section_path.copy(),
                        location=location, order_index=order_index,
                    ))
                    order_index += 1
                    i += 1
                    continue
                # Not a real heading — fall through and treat as a paragraph.
                blocks.append(Block(
                    doc_id=doc_id, block_type="paragraph", text=title,
                    section_path=section_path.copy(),
                    location=location, order_index=order_index,
                ))
                order_index += 1
                i += 1
                continue

            # Table: accumulate consecutive pipe lines into one block.
            if _is_table_line(line):
                table_lines = []
                while i < len(lines) and _is_table_line(lines[i]):
                    table_lines.append(lines[i].rstrip())
                    i += 1
                blocks.append(Block(
                    doc_id=doc_id, block_type="table",
                    text="\n".join(table_lines), section_path=section_path.copy(),
                    location=location, order_index=order_index,
                ))
                order_index += 1
                continue

            # Ordinary paragraph line.
            blocks.append(Block(
                doc_id=doc_id, block_type="paragraph", text=stripped,
                section_path=section_path.copy(),
                location=location, order_index=order_index,
            ))
            order_index += 1
            i += 1

    return blocks


if __name__ == "__main__":  # standalone smoke test
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "samples/FDS_PriceBook_Automation_V0.pdf"
    bs = parse_pdf(p, "A")
    headings = [b for b in bs if b.block_type == "heading"]
    tables = [b for b in bs if b.block_type == "table"]
    non_heading = [b for b in bs if b.block_type != "heading"]
    with_path = [b for b in non_heading if b.section_path]
    print(f"blocks={len(bs)} headings={len(headings)} tables={len(tables)}")
    print(f"non-heading blocks with section_path: "
          f"{len(with_path)}/{len(non_heading)} "
          f"({100*len(with_path)/max(len(non_heading),1):.1f}%)")
    for h in headings[:15]:
        print(f"  L{h.heading_level} {' > '.join(h.section_path)}")
