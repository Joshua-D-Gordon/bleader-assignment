"""Comparison route."""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.controllers import comparison_controller

router = APIRouter(tags=["comparison"])


@router.post("/compare")
def compare(refresh: bool = False):
    return comparison_controller.compare(refresh=refresh)
