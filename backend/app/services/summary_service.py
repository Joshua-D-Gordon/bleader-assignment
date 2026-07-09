"""Executive Summary Generator (bonus).

Consumes the comparison-engine output (DIFF + MISSING items) and asks the LLM to
rank the "Top 10 Most Important Changes" by SEMANTIC IMPORTANCE — business/scope
impact — not document order. Structured Outputs enforce the ranked shape.
"""
from __future__ import annotations

import json

from backend.app import config
from backend.app.clients import openai_client
from backend.app.services.comparison_service import get_comparison

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["changes"],
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rank", "title", "importance", "why_it_matters", "citations"],
                "properties": {
                    "rank": {"type": "integer"},
                    "title": {"type": "string"},
                    "importance": {"type": "string", "enum": ["high", "medium", "low"]},
                    "why_it_matters": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

_SYSTEM = (
    "You are a product analyst. Given a structured list of changes between an older "
    "document (Doc A) and a newer one (Doc B), produce the TOP 10 most important "
    "changes ranked by SEMANTIC IMPORTANCE — business impact, scope, and risk — NOT "
    "by document order. Added capabilities, changed numbers/scope, and removed "
    "content all count. Each item needs a crisp title, an importance level, a one- "
    "to-two sentence rationale, and the source citations it draws from."
)


def _condense(comparison: dict) -> str:
    lines = []
    for d in comparison["diff"]:
        lines.append(f"[CHANGED] {d['sourceA']} -> {d['sourceB']} :: {d['reason']}")
    for m in comparison["missing"]:
        lines.append(f"[ONLY IN {m['source_file']}] {m['location']} :: "
                     f"{m['text'][:300]}")
    return "\n".join(lines)


def top_changes() -> dict:
    comparison = get_comparison()
    payload = _condense(comparison)
    resp = openai_client().chat.completions.create(
        model=config.REASONING_MODEL,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": f"CHANGES:\n{payload}"}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "top_changes", "strict": True, "schema": _SCHEMA},
        },
    )
    data = json.loads(resp.choices[0].message.content)
    data["changes"] = sorted(data["changes"], key=lambda c: c["rank"])[:10]
    return data
