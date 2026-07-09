"""Command-line interface for the four challenge deliverables.

Run from the repo root (no server needed). Ingest first with
`python -m backend.run_ingestion`, then:

  python -m backend.cli compare                 # MATCH/DIFF/MISSING JSON
  python -m backend.cli compare -o out.json     # ...also written to a file

  python -m backend.cli chat A How many currencies are supported?
  python -m backend.cli chat cross What changed in currencies between versions?
  python -m backend.cli chat B                  # interactive (omit the question)

  python -m backend.cli summary                 # Top-10 changes by importance

Tip: the ./run wrapper does all of this without the venv/module prefix, e.g.
  ./run chat A how many currencies are supported
"""
from __future__ import annotations

import argparse
import json
import sys

# ANSI helpers (fall back to plain text if not a TTY).
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s
def _bold(s: str) -> str: return _c("1", s)
def _dim(s: str) -> str: return _c("2", s)
def _cyan(s: str) -> str: return _c("36", s)


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _repl(run_fn, label: str) -> None:
    """Interactive question loop; keeps the process (and agent graph) warm."""
    print(_bold(f"Interactive chat — {label}."), _dim("Type a question, or 'exit' to quit."))
    while True:
        try:
            q = input(_cyan("\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        _print_chat(run_fn(q))


def _print_chat(result: dict) -> None:
    print(_bold("\nAnswer:"))
    print(result["answer"])
    print(_bold("\nCitations:"))
    for c in result["citations"]:
        print(f"  {_cyan('['+c['doc_id']+']')} {c['source']}"
              f"{_dim('  score=' + str(c['score']))}")
    print(_bold("\nRetrieval trace (agent steps):"))
    for i, r in enumerate(result["retrievals"], 1):
        print(f"  {_dim(f'round {i}')} doc={r['doc_id']} "
              f"top_score={r['top_score']} query={r['query']!r}")
    print()


# --- command handlers --------------------------------------------------------
def _cmd_compare(args) -> None:
    from backend.app.services import comparison_service
    result = comparison_service.get_comparison()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.out}  "
              f"(match={len(result['match'])} diff={len(result['diff'])} "
              f"missing={len(result['missing'])})")
    else:
        _print_json(result)


_TARGETS = {"a": "A", "doca": "A", "doc_a": "A",
            "b": "B", "docb": "B", "doc_b": "B",
            "cross": "cross", "both": "cross"}


def _cmd_chat(args) -> None:
    from backend.app.services import agent_service
    target = _TARGETS.get(args.target.lower())
    if target is None:
        sys.exit("target must be A, B, or cross")
    if target == "cross":
        allowed, label = ["A", "B"], "cross-document (Doc A vs Doc B)"
    else:
        allowed, label = [target], f"Doc {target}"
    run_fn = lambda q: agent_service.run(q, allowed_docs=allowed)
    question = " ".join(args.question).strip()   # nargs='*' -> no quoting needed
    if question:
        _print_chat(run_fn(question))
    else:
        _repl(run_fn, label)


def _cmd_summary(args) -> None:
    from backend.app.services import summary_service
    _print_json(summary_service.top_changes())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backend.cli",
        description="Semantic Document Reconciliation — CLI for all four deliverables.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("compare", help="MATCH/DIFF/MISSING comparison JSON")
    pc.add_argument("-o", "--out", help="write JSON to this file instead of stdout")
    pc.set_defaults(func=_cmd_compare)

    pch = sub.add_parser("chat", help="ask Doc A, Doc B, or cross (both); omit question for interactive mode")
    pch.add_argument("target", help="A, B, or cross")
    pch.add_argument("question", nargs="*", help="the question (omit to chat interactively)")
    pch.set_defaults(func=_cmd_chat)

    ps = sub.add_parser("summary", help="Top-10 most important changes, ranked")
    ps.set_defaults(func=_cmd_summary)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
