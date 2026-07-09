"""Document Comparison Engine (Deliverable 1).

Produces the required MATCH / DIFF / MISSING JSON. This is deliberately multi-step
(not a single LLM call):

  1. Parse + chunk both documents (reusing the ingestion parsers).
  2. Build comparable *sections* by grouping chunks under their numbered heading.
  3. Embed every section and greedily align A-sections to B-sections by cosine
     similarity (the two versions are heavily restructured, so title matching
     alone is insufficient — semantic similarity does the real work).
  4. For each aligned pair, an LLM classifies MATCH vs DIFF and explains why,
     using OpenAI Structured Outputs so the classification obeys a fixed schema.
  5. Unaligned sections on either side become MISSING (present in one, absent in
     the other). Assemble and validate against the required output schema.

Results are cached in-process; the first call is the only expensive one, so the
executive-summary endpoint reuses it for free.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor

from jsonschema import validate

from backend.app import config
from backend.app.clients import openai_client
from backend.ingestion.chunker import chunk_blocks
from backend.ingestion.embedder import embed_texts
from backend.ingestion.models import Chunk
from backend.ingestion.parsers.docx_parser import parse_docx
from backend.ingestion.parsers.pdf_parser import parse_pdf

_ALIGN_THRESHOLD = 0.45   # min cosine for two sections to be considered "the same"
_MAX_SECTION_CHARS = 6000

# Required top-level output schema (from the challenge brief).
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["missing", "diff", "match"],
    "properties": {
        "missing": {"type": "array", "items": {
            "type": "object",
            "required": ["text", "source_file", "location"],
            "properties": {"text": {"type": "string"},
                           "source_file": {"type": "string"},
                           "location": {"type": "string"}},
        }},
        "diff": {"type": "array", "items": {
            "type": "object",
            "required": ["docA_text", "docB_text", "reason", "sourceA", "sourceB"],
            "properties": {"docA_text": {"type": "string"},
                           "docB_text": {"type": "string"},
                           "reason": {"type": "string"},
                           "sourceA": {"type": "string"},
                           "sourceB": {"type": "string"}},
        }},
        "match": {"type": "array", "items": {
            "type": "object",
            "required": ["textA", "textB", "source"],
            "properties": {"textA": {"type": "string"},
                           "textB": {"type": "string"},
                           "source": {"type": "string"}},
        }},
    },
}

# Structured-output schema for the per-pair classification step.
_CLASSIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["classification", "reason"],
    "properties": {
        "classification": {"type": "string", "enum": ["match", "diff"]},
        "reason": {"type": "string"},
    },
}


class Section:
    """A comparable unit: a numbered heading plus its aggregated body text."""

    def __init__(self, doc_id: str, title: str, location: str, text: str):
        self.doc_id = doc_id
        self.title = title
        self.location = location
        self.text = text[:_MAX_SECTION_CHARS]

    def citation(self) -> str:
        return f"{config.DOCUMENTS[self.doc_id]['display']} / {self.location} / {self.title}"

    def snippet(self, n: int = 1200) -> str:
        return self.text[:n]


def _section_key(chunk: Chunk) -> tuple[str, str]:
    """Group chunks by their most relevant numbered heading (title, location)."""
    path = chunk.section_path
    if not path:
        return ("(untitled)", chunk.location)
    # Prefer the level-2 numbered heading (e.g. "3. Current Process") if present,
    # else the deepest available heading.
    title = path[1] if len(path) > 1 else path[0]
    return (title, chunk.location)


def _build_sections(doc_id: str) -> list[Section]:
    path = config.sample_path(doc_id)
    parser = parse_pdf if path.suffix.lower() == ".pdf" else parse_docx
    chunks = chunk_blocks(parser(str(path), doc_id))

    grouped: dict[str, dict] = {}
    for c in chunks:
        title, _ = _section_key(c)
        g = grouped.setdefault(title, {"loc": c.location, "parts": []})
        g["parts"].append(c.raw_text)
    return [Section(doc_id, title, g["loc"], "\n\n".join(g["parts"]))
            for title, g in grouped.items()]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _greedy_align(secs_a, vecs_a, secs_b, vecs_b):
    """Return (pairs, unmatched_a_idx, unmatched_b_idx) via greedy 1:1 matching."""
    candidates = []
    for i in range(len(secs_a)):
        for j in range(len(secs_b)):
            s = _cosine(vecs_a[i], vecs_b[j])
            if s >= _ALIGN_THRESHOLD:
                candidates.append((s, i, j))
    candidates.sort(reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for s, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); pairs.append((i, j, s))
    unmatched_a = [i for i in range(len(secs_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(secs_b)) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


_CLASSIFY_SYSTEM = (
    "You compare two versions of the same document section. Decide if they are a "
    "MATCH (identical or semantically equivalent — same meaning, even if reworded) "
    "or a DIFF (meaningfully changed: different facts, numbers, scope, or added/"
    "removed substance). Give a one-sentence reason citing the specific change."
)


def _classify_pair(sec_a: Section, sec_b: Section) -> dict:
    user = (f"SECTION A ({sec_a.title}):\n{sec_a.snippet()}\n\n"
            f"SECTION B ({sec_b.title}):\n{sec_b.snippet()}")
    resp = openai_client().chat.completions.create(
        model=config.REASONING_MODEL,
        messages=[{"role": "system", "content": _CLASSIFY_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "classification", "strict": True,
                            "schema": _CLASSIFY_SCHEMA},
        },
    )
    return json.loads(resp.choices[0].message.content)


def compare_documents() -> dict:
    """Run the full comparison and return the required-schema JSON."""
    secs_a = _build_sections("A")
    secs_b = _build_sections("B")

    vecs_a = embed_texts([s.text for s in secs_a])
    vecs_b = embed_texts([s.text for s in secs_b])

    pairs, unmatched_a, unmatched_b = _greedy_align(secs_a, vecs_a, secs_b, vecs_b)

    # Classify aligned pairs concurrently.
    def classify(pair):
        i, j, _ = pair
        return pair, _classify_pair(secs_a[i], secs_b[j])
    with ThreadPoolExecutor(max_workers=6) as pool:
        classified = list(pool.map(classify, pairs))

    result = {"missing": [], "diff": [], "match": []}
    for (i, j, _score), verdict in classified:
        a, b = secs_a[i], secs_b[j]
        if verdict["classification"] == "match":
            result["match"].append({
                "textA": a.snippet(), "textB": b.snippet(),
                "source": f"{a.citation()}  +  {b.citation()}",
            })
        else:
            result["diff"].append({
                "docA_text": a.snippet(), "docB_text": b.snippet(),
                "reason": verdict["reason"],
                "sourceA": a.citation(), "sourceB": b.citation(),
            })
    for i in unmatched_a:
        a = secs_a[i]
        result["missing"].append({
            "text": a.snippet(), "source_file": config.DOCUMENTS["A"]["display"],
            "location": f"{a.location} / {a.title}",
        })
    for j in unmatched_b:
        b = secs_b[j]
        result["missing"].append({
            "text": b.snippet(), "source_file": config.DOCUMENTS["B"]["display"],
            "location": f"{b.location} / {b.title}",
        })

    validate(instance=result, schema=OUTPUT_SCHEMA)
    return result


# --- in-process cache --------------------------------------------------------
_CACHE: dict | None = None


def get_comparison(refresh: bool = False) -> dict:
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = compare_documents()
    return _CACHE
