"""FastAPI application factory.

Thin entry point: builds the app and mounts the routers. The HTTP layer lives in
`routes/`, orchestration in `controllers/`, and business logic in `services/`
(with the ingestion pipeline in `backend/ingestion/`). Swagger UI at /docs.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app import config
from backend.app.routes import all_routers


def create_app() -> FastAPI:
    app = FastAPI(
        title="Semantic Document Reconciliation",
        description="Compare two FDS versions and query them individually or across versions.",
        version="1.0.0",
    )
    for router in all_routers:
        app.include_router(router)

    # Serve the web UI at / (mounted last so API routes take precedence).
    frontend = config.REPO_ROOT / "frontend"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")
    return app


app = create_app()
