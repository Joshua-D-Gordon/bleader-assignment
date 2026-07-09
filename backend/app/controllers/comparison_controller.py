"""Comparison controller: delegate to the comparison service."""
from __future__ import annotations

from backend.app.services import comparison_service


def compare(refresh: bool = False) -> dict:
    return comparison_service.get_comparison(refresh=refresh)
