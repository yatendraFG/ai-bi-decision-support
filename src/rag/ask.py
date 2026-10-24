"""
Interactive document Q&A.   python -m src.rag.ask

Type a question in English or Hinglish; the answer comes with its source document.
Set ANTHROPIC_API_KEY (and `pip install anthropic`) to let Claude write the answers.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.rag.rag_system import RAG


def main():
    rag = RAG()
    llm = "Claude" if os.environ.get("ANTHROPIC_API_KEY") else "extractive (no LLM key set)"
    print(f"Kaveri Retail document assistant | {len(rag.chunks)} chunks | retriever: {rag.method} | answers: {llm}")
    print("Ask a question (blank line or 'exit' to quit).\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            break
        r = rag.ask(q)
        print(f"\nAnswer: {r['answer']}")
        if r["found"] and len(r["sources"]) > 1:
            print("Also see: " + " | ".join(r["sources"][1:]))
        print()


if __name__ == "__main__":
    main()
