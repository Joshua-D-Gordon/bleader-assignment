"""Chat controller: validate input, delegate to the agentic RAG service.

Both chat modes run through the same LangGraph agent (see agent_service); they
differ only in which documents the agent may search.
"""
from __future__ import annotations

from fastapi import HTTPException

from backend.app.services import agent_service


def chat_single(doc_id: str, question: str, top_k: int = 6) -> dict:
    if doc_id not in ("A", "B"):
        raise HTTPException(status_code=400, detail="doc_id must be 'A' or 'B'")
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return agent_service.run(question, allowed_docs=[doc_id])


def chat_cross(question: str, top_k: int = 5) -> dict:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return agent_service.run(question, allowed_docs=["A", "B"])
