"""Interactive CLI for asking the system-design RAG questions.

    python src/chat.py                       # interactive REPL (remembers context)
    python src/chat.py "How does a CDN work?"   # one-shot question

In the REPL the conversation is remembered, so follow-ups work:
    you> Explain the Google File System architecture
    you> what about its fault tolerance?      # "its" resolves to GFS
Commands: 'reset' clears the conversation; 'exit'/'quit' leaves.
"""
import sys
from typing import List, Tuple

from rag import answer

Turn = Tuple[str, str]


def _print_answer(text: str, sources) -> None:
    print("\n" + text.strip() + "\n")
    seen = []
    for s in sources:
        key = f"{s.source}/{s.path}"
        if key not in seen:
            seen.append(key)
    print("Retrieved from:")
    for key in seen:
        print(f"  - {key}")


def ask_once(question: str) -> None:
    text, sources = answer(question)
    _print_answer(text, sources)


def repl() -> None:
    print(
        "System Design RAG (Gemini). I remember the conversation, so you can ask "
        "follow-ups.\nType 'reset' to start over, 'exit' to quit.\n"
    )
    history: List[Turn] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break
        if q.lower() in {"reset", "clear", "new"}:
            history = []
            print("(conversation reset)\n")
            continue
        text, sources = answer(q, history=history)
        _print_answer(text, sources)
        history.append((q, text))
        print()


def main() -> None:
    if len(sys.argv) > 1:
        ask_once(" ".join(sys.argv[1:]))
    else:
        repl()


if __name__ == "__main__":
    main()
