"""Pinecone index lifecycle + upsert/query helpers.

One serverless index holds both documents. Every vector carries `doc_id` in its
metadata, so single-document retrieval is a metadata filter and cross-document
retrieval is two filtered queries against the same index (see engine/rag.py).
"""
from __future__ import annotations

import time

from ..app import config
from ..app.clients import pinecone_client
from .models import Chunk

# Pinecone metadata has a 40KB/vector cap; keep stored raw_text well under it.
_MAX_META_TEXT = 8000


def ensure_index() -> None:
    pc = pinecone_client()
    existing = {ix["name"] for ix in pc.list_indexes()}
    if config.PINECONE_INDEX in existing:
        return
    from pinecone import ServerlessSpec
    pc.create_index(
        name=config.PINECONE_INDEX,
        dimension=config.EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    # Wait until the index is ready to accept upserts.
    while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
        time.sleep(1)


def _index():
    return pinecone_client().Index(config.PINECONE_INDEX)


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]],
                  batch_size: int = 100) -> int:
    """Upsert chunk vectors + citation metadata. Returns count upserted."""
    items = []
    for chunk, vec in zip(chunks, vectors):
        items.append({
            "id": chunk.chunk_id,
            "values": vec,
            "metadata": {
                "doc_id": chunk.doc_id,
                "display": config.DOCUMENTS[chunk.doc_id]["display"],
                "section_path": chunk.breadcrumb(),
                "location": chunk.location,
                "chunk_type": chunk.chunk_type,
                "order_index": chunk.order_index,
                "raw_text": chunk.raw_text[:_MAX_META_TEXT],
            },
        })
    index = _index()
    for i in range(0, len(items), batch_size):
        index.upsert(vectors=items[i:i + batch_size])
    return len(items)


def clear_doc(doc_id: str) -> None:
    """Delete all vectors for a document so re-ingestion is idempotent."""
    try:
        _index().delete(filter={"doc_id": doc_id})
    except Exception:
        # Fresh/empty index may not support filtered delete yet; ignore.
        pass


def query(vector: list[float], top_k: int = 6,
          doc_id: str | None = None) -> list[dict]:
    """Similarity search, optionally filtered to one document."""
    kwargs = {"vector": vector, "top_k": top_k, "include_metadata": True}
    if doc_id is not None:
        kwargs["filter"] = {"doc_id": doc_id}
    res = _index().query(**kwargs)
    return [
        {
            "score": m["score"],
            "doc_id": m["metadata"]["doc_id"],
            "display": m["metadata"]["display"],
            "section_path": m["metadata"].get("section_path", ""),
            "location": m["metadata"]["location"],
            "chunk_type": m["metadata"].get("chunk_type", "text"),
            "raw_text": m["metadata"]["raw_text"],
        }
        for m in res.get("matches", [])
    ]


def stats() -> dict:
    return _index().describe_index_stats().to_dict()
