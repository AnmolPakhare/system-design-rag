"""Ingest the external reference links (papers, blogs, docs) cited by the
tutorials into the SAME vector store as the repo markdown.

    python src/ingest_web.py                 # crawl all content links
    python src/ingest_web.py --limit 20       # quick test on first 20 links
    python src/ingest_web.py --workers 12      # more concurrent fetches

Resumable: already-embedded chunks are skipped, so re-running continues.
"""
import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import chromadb
from tqdm import tqdm

from chunker import _window
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_PACE_SECONDS,
)
from embeddings import BACKEND, NEEDS_PACING, embed_documents
from fetch_web import fetch
from links import extract_links


def _id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def build_web_index(workers: int, limit: int | None) -> None:
    col = _collection()

    links = list(extract_links().keys())
    if limit:
        links = links[:limit]
    print(f"[web] {len(links)} candidate links; fetching with {workers} workers ...")

    docs = []  # (url, title, text)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, u): u for u in links}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="fetch"):
            url = futs[fut]
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001
                res = None
            if res:
                docs.append((url, res[0], res[1]))
    print(f"[web] fetched {len(docs)} usable pages (of {len(links)})")

    # Window each page into chunks tagged with the source URL/title.
    chunks = []  # (text, metadata)
    for url, title, text in docs:
        dom = urlparse(url).netloc
        for window in _window(text):
            body = f"[web | {dom}] {title}\n{url}\n\n{window}"
            chunks.append((body, {"source": "web", "path": url, "heading": title}))
    print(f"[web] produced {len(chunks)} chunks")

    existing = set(col.get(include=[])["ids"])
    pending = [(t, m) for (t, m) in chunks if _id(t) not in existing]
    print(f"[web] backend={BACKEND}: {len(pending)} new chunks to embed.")

    for start in tqdm(range(0, len(pending), EMBED_BATCH_SIZE), desc="embed"):
        batch = pending[start : start + EMBED_BATCH_SIZE]
        texts = [t for t, _ in batch]
        vectors = embed_documents(texts)
        col.add(
            ids=[_id(t) for t in texts],
            embeddings=vectors,
            documents=texts,
            metadatas=[m for _, m in batch],
        )
        if NEEDS_PACING and start + EMBED_BATCH_SIZE < len(pending):
            time.sleep(EMBED_PACE_SECONDS)

    print(f"[done] collection now holds {col.count()} chunks total.")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest external reference links.")
    p.add_argument("--workers", type=int, default=8, help="Concurrent fetchers.")
    p.add_argument("--limit", type=int, default=None, help="Only first N links (test).")
    args = p.parse_args()
    build_web_index(workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
