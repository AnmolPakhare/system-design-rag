"""Quota-free measurement of the retrieval fixes (no Gemini calls).

Quantifies exactly what the judge flagged ("redundant context") by comparing
plain top-k retrieval against the new diversified retrieve(), on the eval
questions, using only local embeddings:

  - distinct_sources: how many different files/URLs the k chunks came from (higher = more diverse)
  - redundancy: average pairwise cosine similarity among the k chunks (lower = less duplicative)
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from embeddings import embed_query
from evaluate import EVAL_QUESTIONS
from rag import TOP_K, _cosine, _get_collection, retrieve

_col = _get_collection()


def raw_topk(question: str, k: int = TOP_K):
    """Plain similarity top-k with no dedup/diversity (the old behaviour)."""
    v = embed_query(question)
    r = _col.query(query_embeddings=[v], n_results=k,
                   include=["documents", "metadatas"])
    return r["documents"][0], r["metadatas"][0]


def metrics(docs, metas):
    embs = [np.asarray(embed_query(d), dtype=float) for d in docs]
    pairs = list(itertools.combinations(embs, 2))
    redundancy = sum(_cosine(a, b) for a, b in pairs) / len(pairs) if pairs else 0.0
    distinct = len(set(m.get("path", "") for m in metas))
    return distinct, redundancy


def main() -> None:
    print(f"{'':46s}  raw top-k        diversified")
    print(f"{'question':46s}  src  redund    src  redund")
    raw_d = raw_r = mmr_d = mmr_r = 0.0
    n = len(EVAL_QUESTIONS)
    for q in EVAL_QUESTIONS:
        rd, rm = metrics(*raw_topk(q))
        md, mm = metrics(*retrieve(q))
        raw_d += rd; raw_r += rm; mmr_d += md; mmr_r += mm
        print(f"{q[:46]:46s}  {rd:>2d}   {rm:.3f}    {md:>2d}   {mm:.3f}")
    print("-" * 78)
    print(f"{'AVERAGE':46s}  {raw_d/n:.1f}  {raw_r/n:.3f}    {mmr_d/n:.1f}  {mmr_r/n:.3f}")
    print("\n(distinct sources: higher is better; redundancy: lower is better)")


if __name__ == "__main__":
    main()
