"""Root + health routes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.ingestion import pinecone_store

router = APIRouter(tags=["meta"])


@router.get("/health")
def health():
    try:
        stats = pinecone_store.stats()
        return {"status": "ok", "vectors": stats.get("total_vector_count")}
    except Exception as e:  # index not created / keys missing
        return {"status": "degraded", "detail": str(e)}
