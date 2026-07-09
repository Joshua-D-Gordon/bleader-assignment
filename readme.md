# Semantic Document Reconciliation

An AI system that compares two versions of a Functional Design Specification
(one PDF, one DOCX), detects what changed / was added / was removed, and lets you
query either document or ask cross-document questions — all with source citations.

Built for the AI Engineer take-home. Sample documents:
- **Doc A** — `FDS_PriceBook_Automation_V0.pdf` (older baseline)
- **Doc B** — `FDS_PriceBook_Automation_V5.docx` (newer, expanded)

---

## What it does

| Deliverable | Endpoint | Notes |
|---|---|---|
| **Comparison engine** | `POST /compare` | Structured `MATCH` / `DIFF` / `MISSING` JSON with citations + LLM explanations |
| **Single-doc RAG chat** | `POST /chat` | Grounded Q&A over Doc A *or* Doc B, citations required |
| **Cross-doc chat** | `POST /chat/cross` | Comparative Q&A via dual retrieval, cites both sources |
| **Executive summary** | `GET /summary` | "Top 10 changes" ranked by semantic importance |
| Ingestion | `POST /ingest` | Parse → chunk → embed → upsert |
| Health | `GET /health` | Liveness + vector count |

Swagger UI at **`/docs`** is the interactive test surface.

---

## Setup

### Prerequisites
- An **OpenAI API key** (LLM + embeddings) and a **Pinecone API key** (free tier is fine).
- Copy `.env.example` → `.env` and fill both keys.

```bash
cp .env.example .env      # then edit in your keys
```

### Native

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd ..
```

Then use the single **`./run`** entry point (handles the venv + module paths for you):

```bash
python -m backend.run_ingestion  # one-time: parse -> chunk -> embed -> upsert (both docs)

./run help                       # list all commands

# the four deliverables, straight from the terminal — no quotes needed:
./run chat A how many currencies are supported      # single-doc RAG (Doc A)
./run chat B                                         # interactive session over Doc B
./run chat cross what changed in the currencies      # cross-document comparison
./run compare -o comparison.json                     # MATCH/DIFF/MISSING JSON
./run summary                                        # Top-10 changes by importance

./run serve                      # start the server, then open http://localhost:8000/
python -m eval.run_eval          # (optional) retrieval hit-rate check
```

Open **http://localhost:8000/** for the web UI (Chat / Compare / Executive Summary),
or **/docs** for the Swagger API.

### Docker

```bash
./run docker        # builds the image and runs the API on http://localhost:8000
```

Then open <http://localhost:8000/> for the web UI (or `/docs` for Swagger). Vectors
live in Pinecone (cloud), so if you already ingested you don't need to re-ingest;
otherwise `curl -X POST http://localhost:8000/ingest` once.

---

## Project structure

Layered HTTP backend (routes → controllers → services) with the ingestion
pipeline as its own domain package:

```
backend/
  app/
    main.py             # app factory: mounts routers
    config.py           # env, model names, document registry
    clients.py          # lazy OpenAI / Pinecone singletons
    schemas.py          # request DTOs
    routes/             # thin HTTP layer (URL ↔ handler)
    controllers/        # validation + orchestration
    services/           # business logic:
                        #   agent_service     (LangGraph agentic RAG — both chat modes)
                        #   comparison_service (MATCH/DIFF/MISSING engine)
                        #   summary_service    (Top-10 exec summary)
                        #   ingestion_service  (parse→chunk→embed→upsert pipeline)
  ingestion/            # pipeline: parsers, chunker, contextualizer, embedder, store
  cli.py                # argparse CLI behind ./run (chat/compare/summary)
  run_ingestion.py      # one-time ingestion CLI (wraps ingestion_service)
run                     # single entry point: ./run {help,chat,compare,summary,serve,docker}
frontend/index.html     # self-contained web UI (served at / by FastAPI)
eval/                   # retrieval hit-rate harness
samples/                # both FDS documents
Dockerfile
DECISIONS.md            # decisions, trade-offs, bugs we hit, what I'd improve
```

## Architecture

```
PDF ─┐                                                     ┌─ /compare  (align + classify)
     ├─ parse ─ chunk ─ contextualize ─ embed ─ Pinecone ─┼─ /chat        ┐ LangGraph
DOCX─┘ (Block)  (Chunk)  (blurb+text)  (3-large) (1 index) ├─ /chat/cross  ┘ agent (RAG-as-tool)
                                                           └─ /summary  (reuses /compare)
```

### Parsing (`backend/ingestion/parsers/`)
Both parsers emit a shared, format-agnostic `Block` type — nothing downstream knows
whether content came from PDF or DOCX.
- **PDF** — `pymupdf4llm.to_markdown(page_chunks=True)`. Markdown gives headings
  (`#`) and tables (`|`) for free. pymupdf4llm maps font-size to heading depth, so
  body paragraphs leak in as level-5 "headings"; we keep only short, top-level
  markup as real headings and demote the rest, keeping `section_path` clean.
- **DOCX** — `python-docx`, walking the body in true reading order (paragraphs and
  tables interleaved). **Heading detection is the subtle part:** this file's
  headings use *custom* style IDs `heading10/20/30` (whose display names are
  "heading 10/20/30" but whose real levels are 1/2/3) mixed with standard
  `Heading4/5`. Parsing the level from the style *name* would assign level 10 to a
  top-level heading. We read each style's `outlineLvl` from `styles.xml`
  (`level = outlineLvl + 1`), with a static style-ID map as fallback.
- Tables from both are rendered to markdown and kept **whole** (never split).

### Chunking (`chunker.py`)
One chunk per section: a heading plus its body, up to the next heading of
equal-or-shallower level. Sections over ~600 tokens are sub-split with LangChain's
token-aware `RecursiveCharacterTextSplitter`. Tables are always standalone chunks.
Each chunk keeps `raw_text` pristine (this is what gets cited and shown) separate
from the text we embed.

### Contextual Retrieval (`contextualizer.py`)
We implement [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval):
before embedding, a cheap LLM (`gpt-4o-mini`) writes a one-sentence blurb situating
each chunk in its document, and we embed `blurb + breadcrumb + raw_text`. This
gives chunks that only make sense in context (e.g. a bare table) a vector that
carries that context. The blurb is derived by the LLM — nothing document-specific
is hardcoded, so it generalizes to any pair of documents.

### Embedding & vector store
- **Embeddings:** OpenAI `text-embedding-3-large` (3072-dim).
- **Vector DB:** **Pinecone** — one serverless index, cosine. Every vector carries
  `doc_id` (+ `display`, `section_path`, `location`, `chunk_type`, `raw_text`) in
  metadata, which powers both citations and per-document filtering.

### Orchestration — LangGraph agent (`backend/app/services/agent_service.py`)
Both chat modes run through **one LangGraph agent** that treats retrieval as a
**tool** (`search_document(doc_id, query)`, wrapping the embedder + Pinecone). This
is a real multi-step agent loop, not a single retrieve-then-answer call — which
satisfies the challenge's *Orchestration* requirement (and exercises *Tool Use*):

```
plan ─► retrieve (search_document tool) ─► assess (score round) ─► improved?
          ▲                                                          │ yes
          └──────────── refine query (LLM) ◄────────────────────────┘
                                                                     │ no / cap
                                                                     ▼
                                                        synthesize + cite (from union)
```
- **plan / refine** (LLM, `gpt-4o-mini`): turn the question into a search query and,
  each loop, reformulate it — the agentic, LLM-driven part.
- **assess** (deterministic): score each round as the **mean of its top-3 cosine
  scores**. The agent always tries one refinement, then keeps looping only while a
  round beats the best-so-far by a margin; it stops on no improvement or a hard cap
  (`MAX_ATTEMPTS`). This replaced an LLM "is this sufficient?" grader, which was
  subjective and misfired — an objective improvement signal is deterministic and cheaper.
- **synthesize** (LLM, `gpt-4o`): answers strictly from the retrieved passages
  (system prompt forbids outside knowledge), with inline citations.
- Passages **accumulate + dedup across rounds**, so synthesis always sees the best of
  every round; the score signal only decides *whether to loop*, never what we answer from.
- The response includes the **retrieval trace** (each round's doc/query/score) so the
  multi-step behavior is visible in Swagger.

**Cross-document strategy.** A **single shared index with a `doc_id` metadata tag**;
for cross-doc questions the agent's `allowed_docs = ["A","B"]`, so each retrieve step
searches **both documents** (two filtered queries) and synthesis contrasts them.
Rationale: a naive unfiltered top-k lets the denser document (B has ~3× the chunks)
crowd the other out, so a comparative question could retrieve zero passages from A.
Per-document retrieval guarantees balanced context. Single-doc mode is the same agent
with `allowed_docs = [doc_id]`.

### Comparison engine (`backend/app/services/comparison_service.py`) — multi-step
1. Parse + chunk both docs, group chunks into comparable **sections**.
2. Embed every section; **greedily align** A↔B by cosine similarity (the versions
   are heavily restructured, so title matching alone fails — semantics do the work).
3. An LLM classifies each aligned pair **MATCH vs DIFF** with a reason, via OpenAI
   **Structured Outputs** (enforced JSON schema).
4. Unaligned sections become **MISSING**. The assembled result is validated against
   the required output schema with `jsonschema`.

Output shape (exactly as required):
```json
{
  "missing": [{ "text": "...", "source_file": "docA.pdf", "location": "Page 12, Section 3.1" }],
  "diff":    [{ "docA_text": "...", "docB_text": "...", "reason": "...", "sourceA": "...", "sourceB": "..." }],
  "match":   [{ "textA": "...", "textB": "...", "source": "docA.pdf / Page 2 + docB.docx / Page 2" }]
}
```

### Executive summary (`backend/app/services/summary_service.py`)
Consumes the comparison DIFF/MISSING set and asks the LLM for the **Top 10 changes
ranked by semantic importance** (business/scope impact, not document order), via
Structured Outputs. Reuses the cached comparison result.

---

## Model & provider choices

| Role | Choice | Why |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-large` (3072d) | Strong retrieval quality; single provider to key |
| Reasoning (compare / agent synthesis / summary) | `gpt-4o` | Structured Outputs support; solid long-context comparison |
| Agent plan/refine + per-chunk context blurbs | `gpt-4o-mini` | Cheap, fast; used for the high-frequency small calls |
| Orchestration | LangGraph | Explicit agent state machine (plan→retrieve→assess→refine→synthesize) with retrieval as a tool |
| Vector DB | Pinecone (serverless, cosine) | Managed, zero-ops free tier; native metadata filtering for the per-doc retrieval strategy |

All overridable via env vars (see `.env.example` / `backend/app/config.py`).

---

## Known limitations / what I'd improve
- **Section alignment is greedy 1:1.** A section split into two across versions, or
  merged, isn't modeled as a partial match — it shows as one DIFF + one MISSING.
  A many-to-many alignment (or an agentic re-check pass) would be more precise.
- **Comparison granularity is the numbered section.** Fine-grained sentence-level
  diffs within a matched section aren't surfaced (the `reason` describes them in
  prose instead).
- **Table semantics.** Tables are embedded as markdown; a large table's cells all
  share one vector, so row-level retrieval isn't possible.
- **No reranker.** Retrieval is pure vector similarity; adding a cross-encoder /
  Pinecone reranking would sharpen top-k on ambiguous queries.
- **The agent's stop signal is a heuristic.** Using the mean of top-3 cosine scores
  as the "did retrieval improve?" gauge is a proxy — with large section/table chunks
  the scores are low (~0.25) and comparing across reformulated queries is noisy. It's
  deterministic and cheap, but a reranker score or an answerability check would be
  more principled.
- **Cost/latency of `/compare` and `/summary`** scale with section count (one LLM
  call per aligned pair). Results are cached in-process; a persistent cache would
  help across restarts.
- **Chat is single-turn.** The agent is agentic within one question but keeps no
  conversation history; follow-up questions don't resolve against prior turns.
