"""Interactive CLI for asking the system-design RAG questions.

    python src/chat.py                      # interactive REPL
    python src/chat.py "How does a CDN work?"   # one-shot question
"""
import sys

from rag import answer


def ask_once(question: str) -> None:
    text, sources = answer(question)
    print("\n" + text.strip() + "\n")
    seen = []
    for s in sources:
        key = f"{s.source}/{s.path}"
        if key not in seen:
            seen.append(key)
    print("Retrieved from:")
    for key in seen:
        print(f"  - {key}")


def repl() -> None:
    print("System Design RAG (Gemini). Type a question, or 'exit' to quit.\n")
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
        ask_once(q)
        print()


def main() -> None:
    if len(sys.argv) > 1:
        ask_once(" ".join(sys.argv[1:]))
    else:
        repl()


if __name__ == "__main__":
    main()
