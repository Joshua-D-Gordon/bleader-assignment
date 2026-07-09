"""Ingestion controller: validate input, delegate to the ingestion service."""
from __future__ import annotations

from fastapi import HTTPException

from backend.app.services import ingestion_service


def ingest(doc: str | None) -> dict:
    if doc is not None and doc not in ("A", "B"):
        raise HTTPException(status_code=400, detail="doc must be 'A' or 'B'")
    doc_ids = [doc] if doc else None
    counts = ingestion_service.ingest_documents(doc_ids)
    return {"upserted": counts, "total": sum(counts.values())}
