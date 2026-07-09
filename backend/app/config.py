"""Central configuration: env vars, model names, and document registry.

Loaded once at import time. All other modules read constants from here so model
choices and file locations live in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = two levels up from this file (backend/app/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "samples"

# Load .env from repo root if present (native runs). In Docker the vars are
# injected via --env-file, so a missing file is fine.
load_dotenv(REPO_ROOT / ".env")

# --- Provider keys -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# --- Model choices (OpenAI for everything) -----------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))  # native dim of 3-large
REASONING_MODEL = os.getenv("REASONING_MODEL", "gpt-4o")       # compare/chat/summary
CONTEXT_MODEL = os.getenv("CONTEXT_MODEL", "gpt-4o-mini")      # per-chunk blurbs

# --- Pinecone ----------------------------------------------------------------
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fds-comparison")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Document registry -------------------------------------------------------
# doc_id -> (filename, human display name, older/newer role). doc_id is what we
# store in vector metadata and filter on; filename is what we cite.
DOCUMENTS = {
    "A": {
        "filename": "FDS_PriceBook_Automation_V0.pdf",
        "display": "FDS_PriceBook_Automation_V0.pdf",
        "role": "older",
    },
    "B": {
        "filename": "FDS_PriceBook_Automation_V5.docx",
        "display": "FDS_PriceBook_Automation_V5.docx",
        "role": "newer",
    },
}


def sample_path(doc_id: str) -> Path:
    return SAMPLES_DIR / DOCUMENTS[doc_id]["filename"]


def require_keys(*names: str) -> None:
    """Fail fast with a clear message if a required key is unset."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
