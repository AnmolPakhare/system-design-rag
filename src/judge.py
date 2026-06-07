"""LLM-as-judge evaluation of the RAG system using a higher-tier Gemini model.

A stronger model (gemini-2.5-pro) scores each answer against the context that
produced it, on faithfulness/groundedness, relevance, completeness, coherence,
and citation quality, plus retrieval quality. It also returns concrete,
actionable fixes for the RAG pipeline, which drive the iterative improvement.
"""
import json
import re
from typing import Dict, List

from gemini_client import generate

# Judge with the strongest model available on this key, in preference order.
# gemini-2.5-pro and the 2.0 models are blocked on the free tier (quota limit 0).
# We try gemini-2.5-flash (a tier above the generator) first and fall back to
# gemini-2.5-flash-lite if flash's small daily quota is exhausted, so the eval
# can always run. The model that actually scored each answer is recorded.
JUDGE_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
JUDGE_MODEL = JUDGE_MODELS[0]  # preferred / display label

DIMENSIONS = [
    "groundedness",   # claims supported by the retrieved context (no hallucination)
    "relevance",      # answer addresses the question
    "completeness",   # covers the key points the context supports
    "coherence",      # clear, well-structured, non-redundant
    "citations",      # cited sources appropriate, not duplicated/noisy
    "retrieval_quality",  # did retrieval surface the right, non-redundant context
]

_JUDGE_TEMPLATE = """You are a strict, fair evaluator of a Retrieval-Augmented \
Generation (RAG) system for system-design study. You are given a QUESTION, the \
CONTEXT excerpts the system retrieved, and the ANSWER it generated.

Score each dimension from 1 (poor) to 5 (excellent):
- groundedness: every claim in the ANSWER is supported by the CONTEXT; no invented facts.
- relevance: the ANSWER directly addresses the QUESTION.
- completeness: the ANSWER covers the key points that the CONTEXT supports.
- coherence: clear structure, no repetition or contradiction.
- citations: the cited sources are appropriate and not duplicated/noisy.
- retrieval_quality: the CONTEXT actually contained what was needed. Penalise if \
the CONTEXT is redundant (near-duplicate excerpts), off-topic, or in a \
NON-ENGLISH language.

Then list specific ISSUES, and concrete FIXES to the RAG pipeline (retrieval, \
chunking, prompt, citation handling, etc.). Fixes must be actionable engineering \
changes, not restatements of the issue.

Respond with ONLY a JSON object, no prose, in exactly this shape:
{{"groundedness":<1-5>,"relevance":<1-5>,"completeness":<1-5>,"coherence":<1-5>,\
"citations":<1-5>,"retrieval_quality":<1-5>,"overall":<1-5>,\
"issues":["..."],"fixes":["..."]}}

=== QUESTION ===
{question}

=== CONTEXT (what the system retrieved) ===
{context}

=== ANSWER (what the system produced) ===
{answer}
"""


def _parse_json(text: str) -> Dict:
    """Extract the first JSON object from the model output."""
    # Strip code fences if present.
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in judge output: {text[:200]}")
    return json.loads(m.group(0))


def judge(question: str, context: str, answer: str) -> Dict:
    """Score one (question, context, answer) triple.

    Tries each judge model in preference order: the preferred ones fail fast (no
    retry) so an exhausted daily quota rolls over to the fallback immediately;
    the last model is tried with full retries.
    """
    prompt = _JUDGE_TEMPLATE.format(question=question, context=context, answer=answer)
    last_exc = None
    for i, model in enumerate(JUDGE_MODELS):
        is_last = i == len(JUDGE_MODELS) - 1
        try:
            raw = generate(prompt, model=model, max_retries=6 if is_last else 1)
            verdict = _parse_json(raw)
            verdict["_judge_model"] = model
            break
        except Exception as exc:  # noqa: BLE001 - try the next judge model
            last_exc = exc
            if is_last:
                raise
            continue
    else:  # pragma: no cover - loop always breaks or raises
        raise last_exc

    # Normalise: ensure all numeric dims exist.
    for d in DIMENSIONS + ["overall"]:
        verdict.setdefault(d, None)
    verdict.setdefault("issues", [])
    verdict.setdefault("fixes", [])
    return verdict
