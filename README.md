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
- **Generation:** `gemini-2.5-flash` (always Gemini)
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
GEMINI_MODEL=gemini-2.5-flash
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
    ├── links.py          # extract external reference URLs from the tutorials
    ├── fetch_web.py      # fetch + extract text from HTML/PDF links
    ├── ingest_web.py     # crawl external links -> chunk -> embed (resumable)
    ├── rag.py            # retrieve + generate
    └── chat.py           # CLI / REPL
```
