"""Aggregates all route routers for the app factory to include."""
from backend.app.routes import (
    chat_routes,
    comparison_routes,
    health_routes,
    ingestion_routes,
    summary_routes,
)

all_routers = [
    health_routes.router,
    ingestion_routes.router,
    comparison_routes.router,
    chat_routes.router,
    summary_routes.router,
]
