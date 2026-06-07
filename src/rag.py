"""Retrieval-augmented querying over the indexed system-design tutorials."""
from dataclasses import dataclass
from typing import List

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from embeddings import embed_query
from gemini_client import generate


@dataclass
class Source:
    source: str
    path: str
    heading: str


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


def retrieve(question: str, top_k: int = TOP_K):
    collection = _get_collection()
    q_vec = embed_query(question)
    res = collection.query(query_embeddings=[q_vec], n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return docs, metas


def answer(question: str, top_k: int = TOP_K):
    docs, metas = retrieve(question, top_k)

    context_blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        context_blocks.append(f"--- Excerpt {i} ({meta['source']}/{meta['path']}) ---\n{doc}")
    context = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ===\n"
    )
    text = generate(prompt)
    sources = [Source(m["source"], m["path"], m["heading"]) for m in metas]
    return text, sources
