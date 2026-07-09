"""Executive-summary route."""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.controllers import summary_controller

router = APIRouter(tags=["summary"])


@router.get("/summary")
def summary():
    return summary_controller.top_changes()
