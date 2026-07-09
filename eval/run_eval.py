"""Retrieval hit-rate harness.

Runs the queries in test_queries.json against the live Pinecone index and reports,
per expectation, whether an expected substring was retrieved within top_k (and at
what rank). Doubles as the embedding/chunking sanity check and as README evidence.

Usage (from repo root, after ingestion):  python -m eval.run_eval
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.ingestion.embedder import embed_query
from backend.ingestion import pinecone_store

_QUERIES = Path(__file__).parent / "test_queries.json"


def _hit_rank(matches: list[dict], needles: list[str]) -> int | None:
    for rank, m in enumerate(matches, 1):
        text = m["raw_text"].lower()
        if any(n.lower() in text for n in needles):
            return rank
    return None


def main() -> None:
    spec = json.loads(_QUERIES.read_text())
    top_k = spec.get("top_k", 6)

    total = passed = 0
    print(f"Running {len(spec['queries'])} queries (top_k={top_k})\n")
    for q in spec["queries"]:
        vec = embed_query(q["question"])
        print(f"• {q['id']}: {q['question']}")
        for exp in q["expectations"]:
            doc_id = exp["doc_id"]
            matches = pinecone_store.query(vec, top_k=top_k, doc_id=doc_id)
            rank = _hit_rank(matches, exp["any"])
            total += 1
            ok = rank is not None
            passed += int(ok)
            mark = f"HIT @rank {rank}" if ok else "MISS"
            print(f"    [{doc_id}] expect {exp['any']!r:50.50} -> {mark}")
        print()

    pct = 100 * passed / total if total else 0
    print(f"Hit rate: {passed}/{total} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
