import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qa  # noqa: E402
from guardrails import sanitize_question  # noqa: E402
from qa.retrieve import index_records  # noqa: E402
from utils import load_records, repo_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive QA over incident records")
    ap.add_argument("--system", choices=["baseline", "direct", "tool_assisted"], default="tool_assisted")
    ap.add_argument("--records", default=repo_path("data", "golden_set", "parsed_docs", "qa"))
    args = ap.parse_args()

    records = load_records(args.records)
    index, records_by_id = index_records(records)

    print(f"system={args.system} records={len(records)}  (Ctrl-D to quit)")
    while True:
        try:
            question = input("question> ").strip()
        except EOFError:
            break
        if not question:
            continue
        clean, removed = sanitize_question(question)
        for snippet in removed:
            print(f"[guardrail removed: {snippet}]")
        resp = qa.answer(args.system, clean, index, records_by_id)
        print(f"  status: {resp.get('status')}")
        print(f"  answer: {resp.get('answer')}")
        for c in resp.get("citations", []):
            print(f"  cite:   {c.get('incident_id')}")
        print()


if __name__ == "__main__":
    main()
