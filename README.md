# System Design RAG (Gemini)

A Retrieval-Augmented Generation system that reads two system-design tutorials
end to end — **every** sub-document / sublink in each repo — embeds them with
Google Gemini, and answers questions grounded in that material.

Tutorials ingested:
- [donnemartin/system-design-primer](https://github.com/donnemartin/system-design-primer)
- [karanpratapsingh/system-design](https://github.com/karanpratapsingh/system-design)

## How it works

```
git clone repos ─► chunk every .md (heading-aware) ─► Gemini embeddings
        │                                                     │
        └──────────────► Chroma vector store ◄────────────────┘
                                  │
   question ─► embed query ─► top-k retrieve ─► gemini-2.5-flash ─► grounded answer
```

- **Embeddings (retrieval):** pluggable via `EMBEDDING_PROVIDER`
  - `local` *(default)* — Chroma's bundled `all-MiniLM-L6-v2` (ONNX, offline,
    no API quota). Used so ingestion isn't blocked by free-tier limits.
  - `gemini` — `gemini-embedding-001` via the API (subject to free-tier
    100 req/min and daily caps; the ingester paces and resumes automatically).
- **Generation:** `gemini-2.5-flash-lite` (always Gemini; set `GEMINI_MODEL` to
  use a different one, e.g. `gemini-2.5-flash`)
- **Vector store:** Chroma (persisted under `data/chroma/`)
- **Chunking:** splits each markdown file at its headings, then *packs*
  consecutive small sections up to ~4 KB (windowing anything larger with
  overlap) and prepends the heading trail so retrieved context survives.

## Setup

```powershell
cd C:\Users\anmol\PycharmProjects\system-design-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Gemini API key lives in `.env` (already populated):

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash-lite   # lower-tier model w/ higher free daily quota
EMBEDDING_PROVIDER=local          # "local" (default) or "gemini"
EMBEDDING_MODEL=gemini-embedding-001
```

> To use Gemini embeddings instead of the local model, set
> `EMBEDDING_PROVIDER=gemini` and re-run ingestion with `--reset`. On the free
> tier this is rate-limited and capped per day; the ingester paces its batches
> and is resumable (re-run without `--reset` to continue where it stopped).

## Build the knowledge base

Clones both repos and indexes every markdown sub-document:

```powershell
python src/ingest.py            # build (skips clone if repos already present)
python src/ingest.py --reset    # wipe vector store and re-embed
python src/ingest.py --reclone  # delete + re-clone repos, then index
```

### Deep dive: external reference links

The tutorials' "further reading" sections link out to papers, engineering blogs
and docs (e.g. the **GFS**, **MapReduce**, **Spanner**, **Dynamo** papers). To
pull that content into the knowledge base too — not just the link text — crawl
the external links and ingest the fetched HTML/PDF text into the same store:

```powershell
python src/ingest_web.py                 # crawl all content links
python src/ingest_web.py --limit 20       # quick test on the first 20 links
python src/ingest_web.py --workers 12      # more concurrent fetches
```

It fetches each link, extracts readable text (HTML via BeautifulSoup, PDFs via
pypdf), skips media/login-walled hosts, then chunks and embeds the result.
Resumable: already-embedded chunks are skipped on re-runs.

## Ask questions

```powershell
python src/chat.py                                   # interactive REPL
python src/chat.py "Compare SQL vs NoSQL for a feed"  # one-shot
```

Every answer is grounded in retrieved excerpts and lists the source files it
drew from.

### Conversational memory

The REPL **remembers the conversation**, so follow-ups that refer back to
earlier turns work:

```
you> Explain the Google File System architecture
you> what about its fault tolerance?      # "its" resolves to GFS
you> reset                                 # clear the conversation
```

Recent user turns are folded into the retrieval query (a free, no-LLM heuristic)
so references like "its"/"that" still fetch the right material, and the recent
history is passed to the generator for continuity — all while keeping each turn
to a single Gemini generation call.

> **Free-tier note:** generation uses the daily free-tier quota for the chosen
> model (lower-tier models like `gemini-2.5-flash-lite` get a higher daily
> allowance than `gemini-2.5-flash`). Retrieval/embeddings are local and
> unlimited; only the final answer step counts against the quota, which resets
> daily. Switch models anytime via `GEMINI_MODEL` in `.env`.

## Project layout

```
system-design-rag/
├── .env                  # Gemini API key + model config
├── requirements.txt
├── data/
│   ├── repos/            # cloned tutorials (gitignored)
│   └── chroma/           # persisted vector store (gitignored)
└── src/
    ├── config.py         # paths, models, chunking + retrieval params
    ├── gemini_client.py  # Gemini embeddings + generation with retry/backoff
    ├── embeddings.py     # pluggable retrieval backend (local | gemini)
    ├── chunker.py        # heading-aware, section-packing markdown chunking
    ├── ingest.py         # clone -> chunk -> embed -> store (resumable)
    ├── links.py          # extract external reference URLs (skips self-repo/forks)
    ├── fetch_web.py      # fetch + extract text from HTML/PDF links
    ├── ingest_web.py     # crawl external links -> chunk -> embed (resumable)
    ├── prune_translations.py  # drop translated-README + self-repo noise chunks
    ├── rag.py            # diversified retrieve (MMR/dedup) + generate
    ├── judge.py          # LLM-as-judge scoring (gemini-2.5-flash)
    ├── evaluate.py       # run judge over a question set, aggregate scores+fixes
    ├── measure_retrieval.py   # quota-free retrieval diversity metrics
    └── chat.py           # CLI / REPL
```

## Evaluation (LLM-as-judge) and iterative improvement

A stronger model grades the RAG's answers so weaknesses can be found and fixed
systematically.

```powershell
python src/evaluate.py --tag baseline      # answer + judge each eval question
python src/measure_retrieval.py             # quota-free retrieval diversity check
```

- **Generator under test:** `gemini-2.5-flash-lite`
- **Judge:** `gemini-2.5-flash` (one tier above the generator), with automatic
  fallback to `gemini-2.5-flash-lite` if flash's daily quota is exhausted; the
  model that actually scored each answer is recorded.
- The judge returns 1-5 scores for **groundedness, relevance, completeness,
  coherence, citations and retrieval_quality**, plus concrete `issues` and
  `fixes`. `evaluate.py` averages the scores and tallies the most-suggested fixes.

**Example of the loop in action.** The baseline judge scored everything ~5
except `retrieval_quality` (4.4), flagging *"context is highly redundant"* and
recommending a *"deduplication step in the retrieval pipeline."* Applied fixes:

1. **Diversified retrieval** (`rag.py`): fetch a larger candidate pool, then drop
   near-duplicate chunks (cosine >= 0.92) and cap chunks per source.
2. **Pruned translated READMEs** (`README-ja.md`, `README-zh-Hans.md`, ...).
3. **Pruned self-repo crawl noise** — the primer's own GitHub pages re-crawled
   under many anchor URLs.

Result: the store went from 3,209 -> 1,888 clean chunks (-41% redundant/noisy),
and retrieval diversity rose from 3.4 -> 4.2 distinct sources per query with
lower average redundancy.
