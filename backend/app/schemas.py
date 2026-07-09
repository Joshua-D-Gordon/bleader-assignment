"""Request DTOs shared by the routes layer."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    doc: Optional[str] = Field(None, description="'A' or 'B'; omit to ingest both")


class ChatRequest(BaseModel):
    doc_id: str = Field(..., description="'A' or 'B'")
    question: str
    top_k: int = 6


class CrossChatRequest(BaseModel):
    question: str
    top_k: int = 5
