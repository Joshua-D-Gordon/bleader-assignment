# Engineering Decisions & Notes

A record of the key decisions, the trade-offs behind them, the bugs we hit along
the way, and what I'd do with more time. Written to show *why* the system looks
the way it does — not just *what* it does.

---

## Architecture decisions

### Layered backend (routes → controllers → services)
The HTTP layer is thin routers; controllers validate and orchestrate; services hold
business logic; the ingestion pipeline is its own domain package. This keeps each
deliverable in one place (`comparison_service`, `agent_service`, `summary_service`,
`ingestion_service`) and makes the code reviewable. A single `./run` entry point and
a `backend/cli.py` sit on top so everything is runnable without memorising module paths.

### Format-agnostic parsing
Both parsers emit the same `Block` type, so nothing downstream ever branches on
PDF-vs-DOCX. Tables are rendered to Markdown and kept whole; a `section_path`
breadcrumb is carried on every block for citations and for the embedding context.

### Pinecone, one index, `doc_id` metadata
A single serverless index with a `doc_id` tag on every vector. Single-doc retrieval
is a metadata filter; cross-doc is two filtered queries against the same index. Chosen
over a local store (Chroma/FAISS) specifically because it **decouples the data from the
container** — you ingest once, and any container/CLI/process with the key sees the same
index (no volume mounting, no re-ingest on `docker run`).

### Contextual Retrieval (Anthropic's technique, via a cheap model)
Before embedding, a `gpt-4o-mini` call writes a one-sentence blurb situating each chunk
in its document; we embed `blurb + breadcrumb + raw_text`. This gives chunks that only
make sense in context (a bare table, a "Stage 6" row) a vector that carries that context.
The blurb is LLM-derived — nothing about price books is hardcoded — so it generalises.

### Orchestration: LangGraph agent with retrieval-as-a-tool
Both chat modes run through one LangGraph `StateGraph`:
`plan → retrieve (search_document tool) → assess → refine → synthesize`. This satisfies
the "agent loop / multi-step retrieval, not a single LLM call" requirement *and*
exercises Tool Use. The same graph serves single-doc (`allowed_docs=[X]`) and cross-doc
(`allowed_docs=["A","B"]`) — the agent itself decides to search both and contrast them.

### Comparison engine: align → classify → assemble
Sections are embedded and **greedily aligned by cosine** (the two versions are heavily
restructured, so title matching alone fails — V5 renamed and reordered almost everything).
Each aligned pair is classified MATCH/DIFF by the LLM via **Structured Outputs**; unaligned
sections become MISSING. The final object is validated against the exact required schema
with `jsonschema`.

---

## Battles (bugs and course-corrections)

### 1. The DOCX heading trap (caught before it poisoned everything)
V5's headings do **not** use standard `Heading 1/2/3` styles. Probing `word/styles.xml`
showed custom style IDs `heading10` / `heading20` / `heading30` — whose display names are
"heading 10/20/30" but whose true outline levels are **1/2/3** — mixed with real
`Heading4/5`. The obvious approach (`int(style.name.split()[-1])`) would assign level
**10** to a top-level heading and flatten the whole `section_path`, silently corrupting
every citation. Fix: read each style's `outlineLvl` (`level = outlineLvl + 1`) with a
static styleId map as fallback. This was the single highest-leverage catch in the project.

### 2. PDF font-size headings
`pymupdf4llm` maps font size to Markdown heading depth, so body **paragraphs** leaked in
as level-5 "headings" (the whole "AudioCodes publishes…" paragraph became a heading). Fix:
treat only short (`≤120` chars), top-level (`≤ H4`) markup as real headings and demote the
rest to paragraphs. Also filtered `-----` horizontal-rule noise and sub-5-token fragments.

### 3. The agent grader that always said "insufficient"
First cut of the chat agent used an LLM boolean grade ("are these passages sufficient?").
It returned `false` **even on passages that clearly contained the answer**, so every query
ran to the iteration cap and the "smart early-stop" was decorative. Root cause: we were
only showing the grader the first **400 chars** of each passage, and the answer (e.g. the
NA-uplift numbers) lived deeper inside a large table — the grader literally couldn't see it.

We fixed the truncation, but then made a bigger call: **replace the subjective LLM grade
with an objective retrieval-improvement stop.** The agent always tries one refinement, then
keeps looping only while a round's `mean(top-3 cosine)` beats the best-so-far by a margin;
it stops on no improvement or a hard cap. Deterministic, cheaper, and honest — and passages
accumulate + dedup across rounds, so a "worse" refined round never hurts the answer.

### 4. openai 1.x → 2.x
Adding `langchain-openai` upgraded `openai` 1.59 → 2.45 (a major bump). We verified every
existing call (`chat.completions`, `embeddings`, Structured Outputs) still worked under 2.x
and pinned the new versions rather than fighting the resolver.

### 5. Python 3.9 friction
The dev venv is 3.9. `str | None` in a Pydantic model field fails to evaluate on 3.9
(fixed with `Optional`), and LangChain's 3-arg `Annotated[...]` TypedDict form broke its
schema conversion (switched plain `TypedDict`). The Docker image uses 3.11, where neither
bites — but supporting 3.9 locally forced these to be clean.

---

## Honest reflections / what I'd improve

### The embedding model choice was a default, not a finding
We used `text-embedding-3-large` (3072-dim) as a "max quality, one provider" default — but
it was **not empirically justified for these documents.** Two short docs / ~116 chunks don't
need 3072 dimensions, and retrieval scores stayed low (~0.25 cosine even on correct hits),
which points at **chunk granularity** as the bottleneck, not embedding capacity. With more
time I'd run `3-large` vs `3-small` vs `3-large@1024` through the eval harness and pick
empirically — I'd expect `3-small` to match at a fraction of the cost.

### Retrieval is the real weak spot
- **Chunks are too coarse.** Whole sections/tables share one vector, diluting similarity and
  making the agent's score signal noisy. Splitting large tables by row and adding overlap would
  sharpen both retrieval and the stop signal.
- **No reranker.** A cross-encoder / Pinecone rerank over the union of retrieved passages would
  beat raw cosine — and would be a better agent stop signal than mean-top-3 cosine (which, being
  a cross-query comparison of low scores, is admittedly a heuristic).
- **Hybrid search.** Dense-only retrieval is weak on exact tokens (`M9K`, `+25%`, CPN codes);
  BM25 + dense would help.

### Comparison granularity
Alignment is greedy 1:1, so a section split-in-two or merged shows as one DIFF + one MISSING
rather than a partial match. Many-to-many alignment (or an agentic re-check) would be more
precise, and sentence-level diffs within a matched section would beat the prose `reason`.

### Product/robustness
Multi-turn conversation (deliberately out of scope), streaming responses, a persistent cache
for `/compare`/`/summary`, a `pytest` suite (we have smoke scripts + a retrieval eval harness),
LLM-as-judge answer evaluation, and CI running the eval on push.

### The MATCH/MISSING explanation tension
The brief says *each result must include a short LLM explanation*, but the **required output
schema** defines `match`/`missing` with no explanation field. We chose exact-schema compliance
(the explanation lives in DIFF's `reason`). If a reviewer weights the constraint over the schema,
adding an `explanation` field to those items is a one-line change.
