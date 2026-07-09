"""FastAPI application factory.

Thin entry point: builds the app and mounts the routers. The HTTP layer lives in
`routes/`, orchestration in `controllers/`, and business logic in `services/`
(with the ingestion pipeline in `backend/ingestion/`). Swagger UI at /docs.
"""
from __future__ import annotations

from fastapi import FastAPI

from backend.app.routes import all_routers


def create_app() -> FastAPI:
    app = FastAPI(
        title="Semantic Document Reconciliation",
        description="Compare two FDS versions and query them individually or across versions.",
        version="1.0.0",
    )
    for router in all_routers:
        app.include_router(router)
    return app


app = create_app()
