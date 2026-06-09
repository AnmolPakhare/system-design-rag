# Architecture

A Retrieval-Augmented Generation (RAG) system that answers system-design
questions grounded in two tutorials — and **every external paper/blog/doc they
link to** — using Google Gemini for generation and an LLM-as-judge loop for
iterative quality improvement.

Tutorials ingested:
- [donnemartin/system-design-primer](https://github.com/donnemartin/system-design-primer)
- [karanpratapsingh/system-design](https://github.com/karanpratapsingh/system-design)

---

## 1. High-level architecture

```
                          INGESTION (offline, build-time)
  ┌───────────────────────────────────────────────────────────────────────┐
  │  git clone repos ─┐                                                     │
  │                   ├─► chunk (heading-aware, section-packing, ~4 KB)     │
  │  crawl external ──┘            │                                        │
  │  links (HTML+PDF)              ▼                                        │
  │                       local embeddings (all-MiniLM-L6-v2, offline)      │
  │                                │                                        │
  │                                ▼                                        │
  │                       Chroma vector store (persisted, cosine)           │
  └───────────────────────────────────────────────────────────────────────┘

                          QUERY (online, per request)
  ┌───────────────────────────────────────────────────────────────────────┐
  │  question + history ─► condense query (heuristic) ─► embed query        │
  │        │                                               │                │
  │        │                                               ▼                │
  │        │                              retrieve candidates (top_k × 5)    │
  │        │                                               │                │
  │        │                              diversify: drop near-dupes (MMR),  │
  │        │                              cap per source ─► top_k context    │
  │        ▼                                               │                │
  │  build prompt (system + history + context + question)  │                │
  │        │                                               │                │
  │        └──────────────► Gemini gemini-2.5-flash-lite ◄──┘                │
  │                                │                                        │
  │                                ▼                                        │
  │                    grounded answer + cited sources                      │
  └───────────────────────────────────────────────────────────────────────┘

                          EVALUATION (LLM-as-judge loop)
  ┌───────────────────────────────────────────────────────────────────────┐
  │  eval questions ─► RAG answer + retrieved context                       │
  │                          │                                              │
  │                          ▼                                              │
  │      gemini-2.5-flash JUDGE  (fallback: gemini-2.5-flash-lite)          │
  │        scores 1-5 on 6 dimensions + issues[] + fixes[]                  │
  │                          │                                              │
  │                          ▼                                              │
  │      aggregate scores → find weak stage → apply fix → re-measure        │
  └───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Components (`src/`)

| Module | Responsibility |
|---|---|
| `config.py` | Central config (paths, model names, chunk/retrieval params), loaded from `.env`. |
| `gemini_client.py` | Gemini SDK wrapper: embeddings + `generate(prompt, model=...)` with 429/503 retry that honours server retry delays. |
| `embeddings.py` | Pluggable retrieval-embedding backend: `local` (MiniLM, default) or `gemini`. |
| `chunker.py` | Heading-aware, **section-packing** markdown chunking; excludes translated READMEs. |
| `ingest.py` | Clone repos → chunk → embed → store. Resumable via content-hash IDs. |
| `links.py` | Extract external reference URLs from the markdown; skip noise + self-repo links. |
| `fetch_web.py` | Fetch + extract readable text from HTML (BeautifulSoup) and PDF (pypdf). |
| `ingest_web.py` | Concurrent crawl of external links → chunk → embed into the same store. |
| `prune_translations.py` | Selectively delete translated-README and self-repo crawl chunks (no full reset). |
| `rag.py` | Query condensing (history), **diversified retrieval (MMR/dedup)**, prompt build, generation. |
| `chat.py` | CLI / REPL with multi-turn memory and a `reset` command. |
| `judge.py` | LLM-as-judge: scores `(question, context, answer)` with a stronger model. |
| `evaluate.py` | Runs the judge over a question set; aggregates dimension scores and fixes. |
| `measure_retrieval.py` | Quota-free retrieval diversity/redundancy metrics (no LLM calls). |

---

## 3. Data flow

### 3.1 Ingestion — repository markdown
1. `git clone --depth 1` both tutorials into `data/repos/`.
2. Walk every `*.md` (all sub-documents / "sublinks"), skipping `.git`,
   `node_modules`, and non-English translated READMEs (`README-ja.md`, …).
3. **Chunking** (`chunker.py`): split each file at markdown headings, then
   **pack** consecutive small sections together up to ~4 KB (windowing anything
   larger with overlap). Each chunk is prefixed with `[repo | path] heading` so
   retrieved snippets keep their context.
4. Embed each chunk and `add()` to Chroma with metadata `{source, path, heading}`
   and a content-hash ID (so re-runs are idempotent / resumable).

### 3.2 Ingestion — external reference links (the "deep dive")
The tutorials only *link* to papers/blogs (e.g. the GFS PDF). To ingest that
content too:
1. `links.py` extracts unique external URLs (filters badges/images, media/login
   hosts, and self-referential links back to the tutorials' own repos/forks).
2. `ingest_web.py` fetches them concurrently; `fetch_web.py` extracts text from
   HTML and PDFs, skipping pages with too little usable text.
3. Pages are windowed into chunks tagged `source="web", path=<url>` and embedded
   into the **same** Chroma collection.

### 3.3 Query
1. `condense_query()` folds recent user turns into the search query (a free,
   no-LLM heuristic) so follow-ups like *"what about its fault tolerance?"*
   embed near the right material.
2. `retrieve()` embeds the query, pulls a candidate pool (`top_k × 5`), then
   **diversifies**: drops near-duplicates (cosine ≥ 0.92) and caps chunks per
   source, backfilling if over-filtered.
3. The prompt is assembled from a system instruction + recent conversation +
   retrieved context + the current question, and sent to `gemini-2.5-flash-lite`.
4. The answer is returned with a de-duplicated list of cited sources.

---

## 4. Key design decisions (and why)

Most decisions were shaped by the **free-tier quota limits** on the API key.

| Decision | Reason |
|---|---|
| **Local embeddings by default** (`all-MiniLM-L6-v2`) | Gemini embedding has a hard daily cap (and `text-embedding-004` wasn't available on the key; `gemini-embedding-001` is). Local embeddings are offline, free, and unlimited, so ingestion never blocks. Generation stays on Gemini. |
| **Section-packing chunker** | Heading-splitting alone produced ~1,865 tiny chunks (many API calls + redundant retrieval). Packing to ~4 KB cut this to ~298 denser chunks with better context. |
| **Resumable ingestion** (content-hash IDs) | A quota/rate-limit stop mid-run never loses progress; re-running continues. |
| **Generator = `gemini-2.5-flash-lite`** | Lower tier than `gemini-2.5-flash`, with a higher free daily generation quota. Configurable via `GEMINI_MODEL`. |
| **Diversified retrieval (MMR/dedup)** | Direct response to the judge flagging redundant context. |
| **Judge fallback chain** | Preferred judge `gemini-2.5-pro` is blocked on the free tier (quota 0); `gemini-2.5-flash` has a small daily cap. The judge tries flash first, then falls back to flash-lite, recording which model scored each answer. |
| **Selective pruning, not full reset** | A reset would wipe the valuable crawled web/paper chunks; selective deletion removes only the noisy ones. |

---

## 5. Evaluation & iterative improvement (LLM-as-judge)

### How a judgement works
The judge receives `(question, retrieved_context, answer)` — including the exact
context the retriever produced — and scores 1-5 on **groundedness, relevance,
completeness, coherence, citations, retrieval_quality**, plus `overall`, and
returns free-text `issues[]` and `fixes[]`. Output is strict JSON, parsed by
`judge.py`. Seeing the context is what lets it tell **hallucination** (low
groundedness) apart from a **retrieval miss** (low retrieval_quality).

### The loop in practice
1. **Baseline** (`evaluate.py --tag baseline`): overall **4.8/5**; everything ~5
   except **`retrieval_quality` 4.4**. Top issue: *"context is highly redundant"*;
   top fix: *"implement a deduplication step in the retrieval pipeline."*
2. **Fixes applied** (the per-dimension scores localized the problem to
   retrieval, not generation):
   - **Diversified retrieval** in `rag.py` (candidate pool → near-duplicate drop
     → per-source cap).
   - **Pruned translated READMEs** — 107 chunks.
   - **Pruned self-repo crawl noise** — 1,214 chunks (the primer's own GitHub
     page re-crawled under many anchor URLs).
   - Store: **3,209 → 1,888** chunks (−41% redundant/noise).
3. **Measured** (`measure_retrieval.py`, quota-free): distinct sources/query
   **3.4 → 4.2**, average redundancy **0.640 → 0.606**.
4. **Judge re-score** (`evaluate.py --tag improved`): re-runs the same rubric on
   the same questions for a comparable before/after — pending the daily quota
   reset.

---

## 6. Models & quotas (this key)

| Model | Use | Free-tier status observed |
|---|---|---|
| `gemini-2.5-flash-lite` | Generator (default) | ~20 requests/day |
| `gemini-2.5-flash` | Preferred judge | ~20 requests/day |
| `gemini-2.5-pro` | (desired judge) | **blocked — quota 0** |
| `gemini-2.0-flash` | (alt) | **blocked — quota 0** |
| `gemini-embedding-001` | (optional embeddings) | hard daily cap |
| `all-MiniLM-L6-v2` (local) | Embeddings (default) | offline, unlimited |

Generation/judging consume the small per-model daily quotas; **retrieval and
embeddings are local and unlimited.**

---

## 7. Project layout

```
system-design-rag/
├── .env                  # API key + model config (gitignored)
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── data/                 # cloned repos, Chroma store, eval reports (gitignored)
└── src/
    ├── config.py
    ├── gemini_client.py
    ├── embeddings.py
    ├── chunker.py
    ├── ingest.py
    ├── links.py
    ├── fetch_web.py
    ├── ingest_web.py
    ├── prune_translations.py
    ├── rag.py
    ├── chat.py
    ├── judge.py
    ├── evaluate.py
    └── measure_retrieval.py
```

---

## 8. Running it

```powershell
# build the knowledge base
python src/ingest.py                 # repo markdown
python src/ingest_web.py             # external reference links (papers/blogs)

# ask questions (REPL remembers context)
python src/chat.py

# evaluate + measure
python src/evaluate.py --tag baseline
python src/measure_retrieval.py
```

See `README.md` for setup and full command reference.
