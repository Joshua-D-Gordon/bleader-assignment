"""Shared, format-agnostic data model.

Both parsers (PDF and DOCX) emit `Block`s. Nothing downstream of the parsers may
branch on the source format — a Block from a PDF and a Block from a DOCX are
interchangeable. Chunks are the unit we embed and store in Pinecone.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

BlockType = Literal["heading", "paragraph", "table"]


class Block(BaseModel):
    """A single structural element in a document, in reading order."""

    doc_id: str                      # "A" or "B"
    block_type: BlockType
    heading_level: Optional[int] = None   # set only for headings (1-based)
    text: str                        # raw text, or a markdown-rendered table
    section_path: list[str] = Field(default_factory=list)  # heading breadcrumb
    location: str                    # citation anchor, e.g. "Page 7" / "Table 3"
    order_index: int                 # global position in the document


class Chunk(BaseModel):
    """A retrieval unit: one section (or a sub-split of a large section)."""

    chunk_id: str
    doc_id: str
    chunk_type: Literal["text", "table"]
    section_path: list[str] = Field(default_factory=list)
    location: str                    # citation anchor for the chunk
    raw_text: str                    # stored, cited, shown to user — never mutated
    order_index: int
    # Populated during the contextual-retrieval step, then embedded. Kept
    # separate from raw_text so citations always show the original content.
    embedding_text: Optional[str] = None

    def breadcrumb(self) -> str:
        return " > ".join(self.section_path)

    def source_citation(self, display_name: str) -> str:
        """Human-readable citation string, e.g. 'docB.docx / Table 3'."""
        return f"{display_name} / {self.location}"
