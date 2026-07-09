"""Ingestion pipeline service: parse -> chunk -> contextualize -> embed -> upsert.

The single source of truth for ingestion logic, used by both the HTTP layer
(ingestion controller) and the CLI entrypoint (backend/run_ingestion.py).
"""
from __future__ import annotations

from backend.app import config
from backend.ingestion import pinecone_store
from backend.ingestion.chunker import chunk_blocks
from backend.ingestion.contextualizer import contextualize
from backend.ingestion.embedder import embed_texts
from backend.ingestion.parsers.docx_parser import parse_docx
from backend.ingestion.parsers.pdf_parser import parse_pdf

_PARSERS = {".pdf": parse_pdf, ".docx": parse_docx}


def ingest_document(doc_id: str) -> int:
    """Ingest a single document; returns the number of vectors upserted."""
    path = config.sample_path(doc_id)
    parser = _PARSERS[path.suffix.lower()]
    print(f"[{doc_id}] parsing {path.name} ...")
    blocks = parser(str(path), doc_id)
    chunks = chunk_blocks(blocks)
    print(f"[{doc_id}] {len(blocks)} blocks -> {len(chunks)} chunks; "
          f"generating context blurbs + embeddings ...")
    chunks = contextualize(chunks)
    vectors = embed_texts([c.embedding_text or c.raw_text for c in chunks])
    pinecone_store.clear_doc(doc_id)
    n = pinecone_store.upsert_chunks(chunks, vectors)
    print(f"[{doc_id}] upserted {n} vectors")
    return n


def ingest_documents(doc_ids: list[str] | None = None) -> dict[str, int]:
    """Ensure the index exists, then ingest the given documents (default: all)."""
    config.require_keys("OPENAI_API_KEY", "PINECONE_API_KEY")
    pinecone_store.ensure_index()
    doc_ids = doc_ids or list(config.DOCUMENTS.keys())
    return {d: ingest_document(d) for d in doc_ids}
