"""Root + health routes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.ingestion import pinecone_store

router = APIRouter(tags=["meta"])


@router.get("/")
def root():
    return {
        "service": "Semantic Document Reconciliation",
        "docs": "Open /docs for the interactive Swagger UI.",
        "endpoints": {
            "POST /ingest": "parse -> chunk -> embed -> upsert",
            "POST /compare": "MATCH/DIFF/MISSING comparison JSON",
            "POST /chat": "single-document RAG Q&A",
            "POST /chat/cross": "cross-document comparative Q&A",
            "GET /summary": "top-10 changes ranked by importance",
            "GET /health": "liveness + index stats",
        },
    }


@router.get("/health")
def health():
    try:
        stats = pinecone_store.stats()
        return {"status": "ok", "vectors": stats.get("total_vector_count")}
    except Exception as e:  # index not created / keys missing
        return {"status": "degraded", "detail": str(e)}
