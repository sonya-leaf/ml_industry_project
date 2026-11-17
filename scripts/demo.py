import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parsing  # noqa: E402
import qa  # noqa: E402
from guardrails import add_warning, sanitize_question  # noqa: E402
from parsing.extract import extract_document, prepare_chunks  # noqa: E402
from qa.retrieve import index_records  # noqa: E402
from utils import load_records, questions, repo_path, validate  # noqa: E402


def parse_one(raw_path: str, system: str) -> dict:
    chunks, guard = prepare_chunks(extract_document(raw_path), True)
    sid = os.path.basename(raw_path)
    iid = f"PARSING-{os.path.splitext(sid)[0]}"
    return add_warning(parsing.parse(system, chunks, sid, iid), guard)


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end demo on golden-set samples")
    ap.add_argument("--parse-system", choices=["baseline", "direct", "tool_assisted"], default="tool_assisted")
    ap.add_argument("--system", choices=["baseline", "direct", "tool_assisted"], default="tool_assisted")
    ap.add_argument("--doc", default=None, help="one raw .docx/.doc to parse")
    ap.add_argument("--question", default=None, help="question to answer over the archive")
    args = ap.parse_args()

    doc = args.doc or sorted(glob.glob(repo_path("data", "golden_set", "raw_docs", "parsing", "*.docx")))[0]
    rec = parse_one(doc, args.parse_system)
    ok, errors = validate(rec, "incident_record")
    obs = (rec.get("observations") or [{}])[0]
    print(f"== parse [{args.parse_system}] {os.path.basename(doc)} ==")
    print(f"  schema_ok={ok}  errors={errors[:3]}")
    print(f"  occurred_at={rec.get('occurred_at')}  event_type={rec.get('event_type')}")
    print(f"  facility={obs.get('facility')}  fault={(rec.get('fault') or {}).get('description')}")
    print(f"  observations={len(rec.get('observations') or [])}  evidence={len(rec.get('evidence') or [])}")

    records = load_records(repo_path("data", "golden_set", "parsed_docs", "qa"))
    index, records_by_id = index_records(records)
    question = args.question
    if not question:
        question = next(iter(questions().values()))["question"]
    clean, removed = sanitize_question(question)
    resp = qa.answer(args.system, clean, index, records_by_id)
    print(f"== qa [{args.system}] ==")
    for snippet in removed:
        print(f"  [guardrail removed: {snippet}]")
    print(f"  question: {question}")
    print(f"  status: {resp.get('status')}")
    print(f"  answer: {resp.get('answer')}")
    for c in resp.get("citations", []):
        print(f"  cite: {c.get('incident_id')}")


if __name__ == "__main__":
    main()
