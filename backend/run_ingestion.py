"""CLI entrypoint for ingestion. Thin wrapper over the ingestion service.

Run from the repo root:  python -m backend.run_ingestion [--doc A|B]
"""
from __future__ import annotations

import argparse

from backend.app.services import ingestion_service
from backend.ingestion import pinecone_store


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest FDS documents into Pinecone.")
    ap.add_argument("--doc", choices=["A", "B"], help="ingest a single document")
    args = ap.parse_args()

    doc_ids = [args.doc] if args.doc else None
    counts = ingestion_service.ingest_documents(doc_ids)
    total = sum(counts.values())
    print(f"\nDone. {total} vectors upserted across {len(counts)} document(s).")
    print("Index stats:", pinecone_store.stats().get("total_vector_count"))


if __name__ == "__main__":
    main()
