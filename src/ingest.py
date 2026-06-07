"""Ingestion pipeline: clone tutorials -> chunk -> embed -> store in Chroma.

Run this once to build (or rebuild) the knowledge base:

    python src/ingest.py            # incremental: skips clone if repo exists
    python src/ingest.py --reset    # wipe the vector store and re-embed
"""
import argparse
import hashlib
import shutil
import subprocess
import time
from pathlib import Path

import chromadb
from tqdm import tqdm

from chunker import load_repo_chunks
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_PACE_SECONDS,
    REPOS,
    REPOS_DIR,
)
from embeddings import BACKEND, NEEDS_PACING, embed_documents


def _chunk_id(text: str) -> str:
    """Stable id from chunk content so re-runs are idempotent / resumable."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def clone_repos() -> None:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        dest = REPOS_DIR / repo["name"]
        if dest.exists():
            print(f"[clone] {repo['name']} already present, skipping.")
            continue
        print(f"[clone] {repo['url']} -> {dest}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo["url"], str(dest)],
            check=True,
        )


def get_collection(reset: bool):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[reset] deleted existing collection.")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(reset: bool) -> None:
    collection = get_collection(reset)

    all_chunks = []
    for repo in REPOS:
        repo_root = REPOS_DIR / repo["name"]
        chunks = load_repo_chunks(repo_root, repo["name"])
        print(f"[chunk] {repo['name']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # Resumability: skip chunks already embedded in a previous (interrupted) run.
    existing = set(collection.get(include=[])["ids"])
    pending = [c for c in all_chunks if _chunk_id(c.text) not in existing]
    print(
        f"[embed] backend={BACKEND}: {len(all_chunks)} total chunks, "
        f"{len(existing)} already indexed, {len(pending)} to embed."
    )

    batches = range(0, len(pending), EMBED_BATCH_SIZE)
    for bi, start in enumerate(tqdm(batches)):
        batch = pending[start : start + EMBED_BATCH_SIZE]
        texts = [c.text for c in batch]
        vectors = embed_documents(texts)
        collection.add(
            ids=[_chunk_id(c.text) for c in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {"source": c.source, "path": c.path, "heading": c.heading}
                for c in batch
            ],
        )
        # Pace API-based embedding to stay under the free-tier rate limit.
        if NEEDS_PACING and start + EMBED_BATCH_SIZE < len(pending):
            time.sleep(EMBED_PACE_SECONDS)

    print(f"[done] indexed {collection.count()} chunks into {CHROMA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the system-design RAG index.")
    parser.add_argument(
        "--reset", action="store_true", help="Wipe the vector store before indexing."
    )
    parser.add_argument(
        "--reclone",
        action="store_true",
        help="Delete and re-clone the tutorial repos first.",
    )
    args = parser.parse_args()

    if args.reclone and REPOS_DIR.exists():
        shutil.rmtree(REPOS_DIR)

    clone_repos()
    build_index(reset=args.reset)


if __name__ == "__main__":
    main()
