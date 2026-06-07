"""Thin wrapper around the Google Gemini (google-genai) SDK.

Provides batched document/query embeddings and answer generation. On rate-limit
(429) errors it honours the server-provided retry delay so large ingestion runs
survive the free-tier quota windows.
"""
import re
import time
from typing import List

from google import genai
from google.genai import types

from config import EMBEDDING_MODEL, GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)

# Pull "retry in 50.96s" or 'retryDelay': '50s' out of a quota error message.
_RETRY_RE = re.compile(r"retry in ([\d.]+)\s*s|retryDelay'?:?\s*'?(\d+)s", re.IGNORECASE)


def _retry_after(exc: Exception) -> float:
    """Best-effort parse of how long the API asked us to wait, in seconds."""
    m = _RETRY_RE.search(str(exc))
    if m:
        secs = float(m.group(1) or m.group(2))
        return min(secs + 2.0, 65.0)  # cap so we never sleep absurdly long
    return 5.0


def _embed(texts: List[str], task_type: str, max_retries: int = 10) -> List[List[float]]:
    """Embed a batch of texts, waiting out 429 quota windows when hit."""
    for attempt in range(max_retries):
        try:
            resp = _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in resp.embeddings]
        except Exception as exc:  # noqa: BLE001 - surface after retries exhausted
            if attempt == max_retries - 1:
                raise
            wait = _retry_after(exc)
            print(f"  embed retry {attempt + 1}/{max_retries}, waiting {wait:.0f}s ...")
            time.sleep(wait)
    return []


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed one batch of document chunks (caller controls batch size/pacing)."""
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> List[float]:
    """Embed a single search query."""
    return _embed([text], task_type="RETRIEVAL_QUERY")[0]


def generate(prompt: str, max_retries: int = 7) -> str:
    """Generate an answer with the chat model.

    Retries 429 (rate limit) and 503 (transient 'high demand') errors,
    honouring a server retry delay when one is provided.
    """
    delay = 3.0
    for attempt in range(max_retries):
        try:
            resp = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return resp.text or ""
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            wait = max(_retry_after(exc), delay)
            print(f"  generate retry {attempt + 1}/{max_retries}, waiting {wait:.0f}s ...")
            time.sleep(wait)
            delay = min(delay * 2, 30.0)
    return ""
