"""Chat routes: single-document and cross-document."""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.controllers import chat_controller
from backend.app.schemas import ChatRequest, CrossChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(req: ChatRequest):
    return chat_controller.chat_single(req.doc_id, req.question, top_k=req.top_k)


@router.post("/chat/cross")
def chat_cross(req: CrossChatRequest):
    return chat_controller.chat_cross(req.question, top_k=req.top_k)
