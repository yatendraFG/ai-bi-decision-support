"""
Interactive AI BI agent.   python -m src.agent.ask_agent

Ask a business question in English or Hinglish. The agent picks the tools, runs them,
prints a report and saves it to reports/agent/.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.agent.agent import ask, HELP


def main():
    brain = "Claude" if os.environ.get("ANTHROPIC_API_KEY") else "offline planner"
    print(f"Kaveri Retail AI BI agent ({brain}). Ask a question; blank line or 'exit' to quit.\n")
    print(HELP.split("\n", 1)[1] + "\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            break
        r = ask(q)
        print("\n" + r["report"].replace("<sub>", "").replace("</sub>", ""))
        if r.get("path"):
            print(f"\n[report saved: {r['path']}]")
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
