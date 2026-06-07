"""Retrieval-augmented querying over the indexed system-design tutorials.

Supports multi-turn conversations: prior turns are used both to rewrite a
follow-up into a standalone search query (so retrieval works on references like
"its" / "that") and to give the generator continuity.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chromadb
import numpy as np

from config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from embeddings import embed_query
from gemini_client import generate

# Retrieval diversification (added after judge feedback flagged redundant context).
CANDIDATE_MULTIPLIER = 5    # fetch this many * top_k candidates, then diversify
DUP_COSINE_THRESHOLD = 0.92  # drop a candidate this similar to one already kept
MAX_PER_SOURCE = 2          # at most this many chunks from the same file/url

# A conversation turn: (user_question, assistant_answer).
Turn = Tuple[str, str]

# How many recent turns to feed into condensing / generation.
HISTORY_TURNS = 6


@dataclass
class Source:
    source: str
    path: str
    heading: str


def _format_history(history: List[Turn]) -> str:
    recent = history[-HISTORY_TURNS:]
    lines = []
    for user, bot in recent:
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {bot}")
    return "\n".join(lines)


# How many recent user turns to fold into a follow-up's retrieval query.
CONDENSE_USER_TURNS = 2


def condense_query(question: str, history: List[Turn]) -> str:
    """Build a standalone retrieval query for a follow-up.

    Uses a free heuristic instead of an LLM call: the recent user questions are
    prepended to the current one so references like "its" / "that" still embed
    near the right material. This keeps each turn to a single Gemini generation
    call (important on the free tier's small daily generate quota).
    """
    if not history:
        return question
    recent_user = [u for u, _ in history[-CONDENSE_USER_TURNS:]]
    return " ".join(recent_user + [question])


SYSTEM_PROMPT = """You are a senior system-design interviewer and mentor. \
Answer the user's question using ONLY the provided context excerpts drawn from \
two system-design tutorials (Donne Martin's system-design-primer and Karan \
Pratap Singh's system-design). Be precise and practical.

Rules:
- Ground every claim in the context. If the context is insufficient, say so \
explicitly rather than inventing details.
- Prefer concrete trade-offs, numbers, and examples when the context offers them.
- Structure longer answers with short headings or bullets.
- End with a "Sources" list of the file paths you drew from.
"""


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def retrieve(question: str, top_k: int = TOP_K):
    """Retrieve top_k chunks, diversified to avoid near-duplicate context.

    Fetches a larger candidate pool (already ranked by similarity), then greedily
    keeps candidates that are not near-duplicates of ones already kept and caps
    how many come from any single source file/URL. This removes the redundant
    context the judge flagged without losing relevance.
    """
    collection = _get_collection()
    q_vec = embed_query(question)
    pool = max(top_k * CANDIDATE_MULTIPLIER, top_k)
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=pool,
        include=["documents", "metadatas", "embeddings"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    embs = [np.asarray(e, dtype=float) for e in res["embeddings"][0]]

    kept, kept_embs, per_source = [], [], {}
    for i in range(len(docs)):
        src = metas[i].get("path", "")
        if per_source.get(src, 0) >= MAX_PER_SOURCE:
            continue
        if any(_cosine(embs[i], ke) >= DUP_COSINE_THRESHOLD for ke in kept_embs):
            continue
        kept.append(i)
        kept_embs.append(embs[i])
        per_source[src] = per_source.get(src, 0) + 1
        if len(kept) >= top_k:
            break

    # Backfill from the pool if diversification filtered out too much.
    if len(kept) < top_k:
        for i in range(len(docs)):
            if i not in kept:
                kept.append(i)
                if len(kept) >= top_k:
                    break

    return [docs[i] for i in kept], [metas[i] for i in kept]


def run(question: str, history: Optional[List[Turn]] = None, top_k: int = TOP_K) -> dict:
    """Full RAG pass. Returns the answer plus the retrieval internals so callers
    (e.g. the evaluation judge) can inspect the context that grounded it."""
    history = history or []

    # 1) Resolve references against history so retrieval has a standalone query.
    search_query = condense_query(question, history)
    docs, metas = retrieve(search_query, top_k)

    context_blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        context_blocks.append(f"--- Excerpt {i} ({meta['source']}/{meta['path']}) ---\n{doc}")
    context = "\n\n".join(context_blocks)

    # 2) Include recent conversation so the answer stays coherent across turns.
    history_block = ""
    if history:
        history_block = f"=== CONVERSATION SO FAR ===\n{_format_history(history)}\n\n"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{history_block}"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== CURRENT QUESTION ===\n{question}\n\n"
        f"=== ANSWER ===\n"
    )
    text = generate(prompt)
    sources = [Source(m["source"], m["path"], m["heading"]) for m in metas]
    return {
        "question": question,
        "search_query": search_query,
        "answer": text,
        "sources": sources,
        "context": context,
        "docs": docs,
        "metas": metas,
    }


def answer(question: str, history: Optional[List[Turn]] = None, top_k: int = TOP_K):
    result = run(question, history, top_k)
    return result["answer"], result["sources"]
