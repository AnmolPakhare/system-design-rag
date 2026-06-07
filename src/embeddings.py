"""Pluggable embedding backend for retrieval.

Selected via EMBEDDING_PROVIDER in config:
  - "local"  : Chroma's bundled all-MiniLM-L6-v2 (ONNX, offline, no quota)
  - "gemini" : gemini-embedding-001 through the Gemini API

Generation always uses Gemini; this only governs the retrieval vectors.
"""
from typing import List

from config import EMBEDDING_PROVIDER

if EMBEDDING_PROVIDER == "gemini":
    from gemini_client import embed_documents as _embed_docs
    from gemini_client import embed_query as _embed_query

    # Gemini's free tier is rate-limited, so the ingester paces its batches.
    NEEDS_PACING = True
    BACKEND = "gemini (gemini-embedding-001)"
else:
    from chromadb.utils import embedding_functions

    _ef = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2

    def _embed_docs(texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in _ef(texts)]

    def _embed_query(text: str) -> List[float]:
        return list(map(float, _ef([text])[0]))

    NEEDS_PACING = False
    BACKEND = "local (all-MiniLM-L6-v2)"


def embed_documents(texts: List[str]) -> List[List[float]]:
    return _embed_docs(texts)


def embed_query(text: str) -> List[float]:
    return _embed_query(text)
