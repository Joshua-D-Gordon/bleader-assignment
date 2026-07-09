"""Agentic RAG over the two documents, built as a LangGraph state machine.

This is the orchestration layer for the chat deliverables. Instead of a single
retrieve-then-answer call, an agent drives a real multi-step loop:

    plan ─► retrieve (search_document tool) ─► grade
              ▲                                  │ weak (and attempts left)
              └────────── refine query ◄─────────┘
                                                 │ good / out of attempts
                                                 ▼
                                            synthesize (+ citations)

`search_document` is a LangChain tool wrapping the existing embedding + Pinecone
retrieval — i.e. the RAG retriever exposed as a tool. The graph decides how many
times to retrieve and whether to reformulate, so "a single LLM call per query" is
definitively not what happens here.

Both chat modes use the SAME graph, differing only in `allowed_docs`:
  * single-document  -> allowed_docs = [doc_id]
  * cross-document   -> allowed_docs = ["A", "B"]  (agent retrieves from both,
                        contrasts them, and highlights what changed)

Answers are grounded strictly in tool output; the synthesis prompt forbids
outside knowledge and requires citations drawn from passage metadata.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from backend.app import config
from backend.ingestion import pinecone_store
from backend.ingestion.embedder import embed_query

SEARCH_K = 5
MAX_ATTEMPTS = 3       # hard cap on retrieval rounds
SCORE_MARGIN = 0.02    # a refined round must beat the best-so-far by this to continue

# Cheap model for plan/refine, stronger model for the final synthesis.
_fast = ChatOpenAI(model=config.CONTEXT_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)
_smart = ChatOpenAI(model=config.REASONING_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)


# --- retrieval tool ----------------------------------------------------------
@tool
def search_document(doc_id: str, query: str) -> list[dict]:
    """Search one document (doc_id 'A' or 'B') for passages relevant to `query`.

    Returns a list of passages with their text, score, and source citation.
    """
    vec = embed_query(query)
    return pinecone_store.query(vec, top_k=SEARCH_K, doc_id=doc_id)


# --- graph state -------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    allowed_docs: list[str]
    query: str
    passages: list[dict]        # accumulated + deduped across rounds
    attempts: int
    answer: str
    retrievals: list[dict]      # trace: which (doc_id, query, score) searches ran
    best_score: float           # best round score seen so far
    last_score: float           # score of the most recent round
    decision: str               # "refine" | "stop", read by the router


# --- structured outputs (TypedDict -> returns plain dicts) --------------------
class _Query(TypedDict):
    # a focused keyword search query
    search_query: str


def _passage_key(p: dict) -> tuple:
    return (p["doc_id"], p["location"], p["raw_text"][:40])


def _dedup(passages: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in passages:
        k = _passage_key(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _round_score(round_results: list[dict]) -> float:
    """Retrieval quality of a round = mean of its top-3 cosine scores."""
    scores = sorted((r["score"] for r in round_results), reverse=True)[:3]
    return sum(scores) / len(scores) if scores else 0.0


# --- nodes -------------------------------------------------------------------
def _plan(state: AgentState) -> dict:
    """Turn the natural-language question into an initial focused retrieval query."""
    prompt = (
        "Rewrite the user's question into a concise search query (keywords, no "
        "punctuation fluff) for retrieving passages from a technical document.\n\n"
        f"Question: {state['question']}"
    )
    plan = _fast.with_structured_output(_Query).invoke(prompt)
    return {"query": plan.get("search_query") or state["question"], "attempts": 0,
            "passages": [], "retrievals": [], "best_score": -1.0}


def _retrieve(state: AgentState) -> dict:
    """Search every allowed doc with the current query; score this round."""
    passages = list(state["passages"])
    retrievals = list(state["retrievals"])
    round_results: list[dict] = []
    for doc_id in state["allowed_docs"]:
        results = search_document.invoke({"doc_id": doc_id, "query": state["query"]})
        round_results.extend(results)
        passages.extend(results)
        retrievals.append({"doc_id": doc_id, "query": state["query"],
                           "hits": len(results),
                           "top_score": round(results[0]["score"], 4) if results else 0.0})
    return {"passages": _dedup(passages), "retrievals": retrievals,
            "attempts": state["attempts"] + 1, "last_score": _round_score(round_results)}


def _assess(state: AgentState) -> dict:
    """Decide whether to refine again, based on retrieval-score improvement.

    Always try at least one refinement (round 1 -> refine). After that, keep
    refining only while a round beats the best score so far by SCORE_MARGIN; stop
    on no improvement or when the attempt cap is hit. `best_score` is updated here
    (routers can't mutate state).
    """
    last, best, attempts = state["last_score"], state["best_score"], state["attempts"]
    if attempts >= MAX_ATTEMPTS:
        decision = "stop"
    elif attempts == 1:
        decision = "refine"                       # always try to improve at least once
    else:
        decision = "refine" if last > best + SCORE_MARGIN else "stop"
    return {"best_score": max(best, last), "decision": decision}


def _refine(state: AgentState) -> dict:
    """LLM reformulates the search query to try to retrieve better passages."""
    prompt = (
        "The previous search query did not retrieve clearly better passages. Write a "
        "DIFFERENT, improved search query (keywords, alternative terminology) for the "
        "same information need. Return only the query.\n\n"
        f"Question: {state['question']}\nPrevious query: {state['query']}"
    )
    out = _fast.with_structured_output(_Query).invoke(prompt)
    return {"query": out.get("search_query") or state["query"]}


def _route_after_assess(state: dict) -> str:
    return "refine" if state["decision"] == "refine" else "synthesize"


def _synthesize(state: AgentState) -> dict:
    cross = len(state["allowed_docs"]) > 1
    labeled = []
    for p in state["passages"]:
        cite = f"{p['display']} / {p['location']}"
        if p.get("section_path"):
            cite += f" / {p['section_path']}"
        labeled.append(f"[{p['doc_id']}] ({cite})\n{p['raw_text']}")
    context = "\n\n".join(labeled) or "(no passages retrieved)"

    if cross:
        system = (
            "You answer comparative questions across two versions of a document "
            "(Doc A = older, Doc B = newer) using ONLY the provided passages. Do not "
            "use outside knowledge. Explicitly contrast what each version says and "
            "call out what changed, was added, or removed. Cite sources inline like "
            "(Doc B / Table 3). If the passages don't cover it, say so."
        )
    else:
        system = (
            "You answer questions about a single document using ONLY the provided "
            "passages. Do not use outside knowledge. Cite sources inline like "
            "(FILE / LOCATION). If the answer isn't in the passages, say so plainly."
        )
    msg = _smart.invoke([
        ("system", system),
        ("human", f"PASSAGES:\n{context}\n\nQUESTION: {state['question']}"),
    ])
    return {"answer": msg.content}


# --- graph assembly ----------------------------------------------------------
def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", _plan)
    g.add_node("retrieve", _retrieve)
    g.add_node("assess", _assess)
    g.add_node("refine", _refine)
    g.add_node("synthesize", _synthesize)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "assess")
    g.add_conditional_edges("assess", _route_after_assess,
                            {"refine": "refine", "synthesize": "synthesize"})
    g.add_edge("refine", "retrieve")
    g.add_edge("synthesize", END)
    return g.compile()


_GRAPH = _build_graph()


def _citations(passages: list[dict]) -> list[dict]:
    return [{
        "doc_id": p["doc_id"],
        "source": f"{p['display']} / {p['location']}",
        "section_path": p.get("section_path", ""),
        "score": round(p["score"], 4),
    } for p in passages]


def run(question: str, allowed_docs: list[str]) -> dict:
    """Execute the agent for a question over the given documents."""
    final = _GRAPH.invoke({
        "question": question, "allowed_docs": allowed_docs,
        "query": question, "passages": [], "attempts": 0,
        "answer": "", "retrievals": [], "best_score": -1.0,
        "last_score": 0.0, "decision": "",
    })
    return {
        "answer": final["answer"],
        "citations": _citations(final["passages"]),
        "mode": "cross" if len(allowed_docs) > 1 else "single",
        "retrievals": final["retrievals"],  # visible multi-step trace
    }
