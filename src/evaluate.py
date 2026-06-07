"""Run the LLM-as-judge evaluation over a fixed question set and report scores.

    python src/evaluate.py            # full eval, saves a JSON report
    python src/evaluate.py --tag baseline

Each question is answered by the RAG (generator model) and then scored by the
judge model (gemini-2.5-pro). Aggregated dimension scores and the most common
suggested fixes are printed so they can drive iterative improvements.
"""
import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from config import DATA_DIR, GEMINI_MODEL
from judge import DIMENSIONS, JUDGE_MODEL, judge
from rag import run

# Representative questions spanning repo markdown and crawled web/paper content.
EVAL_QUESTIONS = [
    "What is the difference between horizontal and vertical scaling, and the trade-offs?",
    "Explain the Google File System (GFS) architecture: master vs chunkservers, "
    "chunk size, and replication / fault tolerance.",
    "When should I use a message queue, and what problems does it solve?",
    "Compare SQL and NoSQL databases and explain when to use each.",
    "What is consistent hashing and why is it used in distributed systems?",
]


def evaluate(tag: str) -> dict:
    print(f"Generator: {GEMINI_MODEL}   |   Judge: {JUDGE_MODEL}")
    print(f"Evaluating {len(EVAL_QUESTIONS)} questions ...\n")

    records = []
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {q[:70]}...")
        result = run(q)
        verdict = judge(q, result["context"], result["answer"])
        records.append(
            {
                "question": q,
                "answer": result["answer"],
                "sources": [f"{s.source}/{s.path}" for s in result["sources"]],
                "verdict": verdict,
            }
        )
        scores = {d: verdict.get(d) for d in DIMENSIONS}
        print(f"      overall={verdict.get('overall')}  {scores}")
        time.sleep(2)  # gentle pacing for the judge model

    # Aggregate dimension averages.
    agg = {}
    for d in DIMENSIONS + ["overall"]:
        vals = [r["verdict"].get(d) for r in records if isinstance(r["verdict"].get(d), (int, float))]
        agg[d] = round(sum(vals) / len(vals), 2) if vals else None

    fixes = Counter()
    issues = Counter()
    for r in records:
        for f in r["verdict"].get("fixes", []):
            fixes[f.strip()] += 1
        for s in r["verdict"].get("issues", []):
            issues[s.strip()] += 1

    judges_used = Counter(r["verdict"].get("_judge_model") for r in records)
    print("\n==================== AGGREGATE ====================")
    print(f"  judge models used: {dict(judges_used)}")
    for d in DIMENSIONS + ["overall"]:
        print(f"  {d:18s}: {agg[d]}")
    print("\n---- Top suggested fixes ----")
    for f, c in fixes.most_common(10):
        print(f"  ({c}x) {f}")
    print("\n---- Top issues ----")
    for s, c in issues.most_common(10):
        print(f"  ({c}x) {s}")

    report = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "generator_model": GEMINI_MODEL,
        "judge_model": JUDGE_MODEL,
        "aggregate": agg,
        "records": records,
        "top_fixes": fixes.most_common(10),
        "top_issues": issues.most_common(10),
    }
    out = Path(DATA_DIR) / f"eval_{tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report -> {out}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Run the LLM-as-judge RAG evaluation.")
    p.add_argument("--tag", default="run", help="Label for this eval run.")
    args = p.parse_args()
    evaluate(args.tag)


if __name__ == "__main__":
    main()
