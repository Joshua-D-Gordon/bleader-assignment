"""Summary controller: delegate to the executive-summary service."""
from __future__ import annotations

from backend.app.services import summary_service


def top_changes() -> dict:
    return summary_service.top_changes()
