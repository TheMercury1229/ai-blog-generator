import sys
from src.app import app


def main():
    print("=" * 60)
    print(" 🚀 Autonomous Technical Blog Generator (LangGraph Multi-Agent)")
    print("=" * 60)

    # Allow topic via CLI argument or interactive prompt
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:]).strip()
    else:
        topic = input("\nEnter blog topic (or press Enter for default 'RAG 101 from noob to pro'): ").strip()
        if not topic:
            topic = "RAG 101 from noob to pro"

    print(f"\n[+] Target Topic: '{topic}'")
    print("[*] Running multi-agent pipeline (Router -> Research -> Orchestrator -> Parallel Workers -> Reducer)...")

    out = app.invoke({"topic": topic, "sections": []})

    print("\n" + "=" * 60)
    print(" ✅ Blog Generation Complete!")
    print("=" * 60)
    print(f"\nPreview of generated blog:\n")
    print(out["final"][:600] + "\n\n... [Full markdown saved in /output directory] ...")


if __name__ == "__main__":
    main()
