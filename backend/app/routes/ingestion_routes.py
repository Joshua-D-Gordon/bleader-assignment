"""Ingestion route."""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.controllers import ingestion_controller
from backend.app.schemas import IngestRequest

router = APIRouter(tags=["ingestion"])


@router.post("/ingest")
def ingest(req: IngestRequest):
    return ingestion_controller.ingest(req.doc)
