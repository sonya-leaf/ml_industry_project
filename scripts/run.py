import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parsing  # noqa: E402
import qa  # noqa: E402
from guardrails import add_warning, sanitize_question  # noqa: E402
from parsing.extract import extract_document, prepare_chunks  # noqa: E402
from qa.retrieve import index_records  # noqa: E402
from utils import load_records, read_csv_rows, read_json, repo_path, write_json  # noqa: E402

DATA = repo_path("data", "golden_set")


# ---- offline mock LLM ------------------------------------------------------


def _mock_complete(system, user, **kw):
    ver = kw.get("system_version", "")
    if ver.startswith("qa.reformulate"):
        q = user.split("Question:", 1)[-1].split("\n", 1)[0].strip() or "вопрос"
        return json.dumps([q])
    if ver.startswith("qa.tool_assisted.agent"):
        if "Tool query_incidents result" not in user:
            q = user.split("Question:", 1)[-1].split("\n", 1)[0].strip() or "вопрос"
            return json.dumps({"action": "query_incidents",
                               "action_input": {"queries": [q], "top_k": 5}})
        # cite the top record the real retriever returned (tool result is JSON [[id, score], ...])
        m = re.search(r'\[\["?([A-Z]+-?\d+)', user)
        iid = m.group(1) if m else None
        if not iid:
            return '{"schema_version":"1.0","status":"insufficient_evidence","answer":"нет данных","citations":[]}'
        return json.dumps({
            "schema_version": "1.0", "status": "answered",
            "answer": f"[{iid}] по данным записи",
            "citations": [{"incident_id": iid, "evidence": {"field_paths": [],
                           "chunk": {"source_file_id": iid, "section": "", "page": None, "text": ""}}}],
        }, ensure_ascii=False)
    if ver.startswith("qa."):
        m = re.search(r"QA[-_ ]?(\d+)", user)
        iid = f"QA-{m.group(1)}" if m else None
        if not iid:
            return '{"schema_version":"1.0","status":"insufficient_evidence","answer":"нет данных","citations":[]}'
        return json.dumps({
            "schema_version": "1.0", "status": "answered",
            "answer": f"[{iid}] по данным записи",
            "citations": [{"incident_id": iid, "evidence": {"field_paths": [],
                           "chunk": {"source_file_id": iid, "section": "", "page": None, "text": ""}}}],
        }, ensure_ascii=False)
    # parsing: minimal valid record (build_record_from_output fills schema/ids)
    return json.dumps({"report_stage": "unknown", "occurred_at": None, "event_type": "unknown",
                       "fault": None, "observations": [], "evidence": []})


def _apply_mock() -> None:
    import qa.baseline
    import qa.direct
    import qa.tool_assisted
    import parsing.direct
    import parsing.tool_assisted
    for mod in (qa.baseline, qa.direct, qa.tool_assisted, parsing.direct, parsing.tool_assisted):
        mod.complete = _mock_complete
    print("[mock] using offline LLM stub (not a real model)")


# ---- parsing ---------------------------------------------------------------


def _incident_id(sid: str, set_name: str) -> str:
    stem = os.path.splitext(sid)[0]
    gold = os.path.join(DATA, "parsed_docs", set_name, stem + ".json")
    if os.path.exists(gold):
        return read_json(gold)["incident_id"]
    return f"{set_name.upper()}-{stem}"


def _parse_docs(raw_dir: str, system: str, set_name: str, guard: bool = True) -> dict:
    preds = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*"))):
        sid = os.path.basename(path)
        iid = _incident_id(sid, set_name)
        chunks, guard_hits = prepare_chunks(extract_document(path), guard)
        preds[sid] = add_warning(parsing.parse(system, chunks, sid, iid), guard_hits)
    return preds


def cmd_parse(args) -> None:
    raw_dir = os.path.join(DATA, "raw_docs", args.set)
    preds = _parse_docs(raw_dir, args.system, args.set, guard=not args.no_guard)
    for sid, rec in preds.items():
        write_json(os.path.join(args.out, sid + ".json"), rec)
    print(f"[parse] set={args.set} system={args.system} produced={len(preds)}")

# ---- QA --------------------------------------------------------------------


def _run_qa(questions, records, system, guard: bool = True) -> dict:
    index, records_by_id = index_records(records)
    responses = {}
    for qid, question in questions:
        if guard:
            question, _ = sanitize_question(question)
        responses[qid] = qa.answer(system, question, index, records_by_id, source_id=qid)
    return responses


def _qa_questions() -> list[tuple[str, str]]:
    rows = read_csv_rows(os.path.join(DATA, "questions", "qa.csv"))
    return [(r["question_id"], r["question"]) for r in rows]


def cmd_qa(args) -> None:
    records = load_records(args.records)
    if not records:
        sys.exit(f"no IncidentRecord *.json in {args.records!r}; "
                 f"run `run.py parse --set qa --out {args.records}` first")
    responses = _run_qa(_qa_questions(), records, args.system, guard=not args.no_guard)
    for qid, resp in responses.items():
        write_json(os.path.join(args.out, qid + ".json"), resp)
    print(f"[qa] system={args.system} records={args.records} produced={len(responses)}")


def cmd_qa_e2e(args) -> None:
    # parse the qa raw docs with the chosen parsing system, then answer
    raw_dir = os.path.join(DATA, "raw_docs", "qa")
    guard = not args.no_guard
    records = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*"))):
        sid = os.path.basename(path)
        iid = _incident_id(sid, "qa")
        chunks, guard_hits = prepare_chunks(extract_document(path), guard)
        records.append(add_warning(parsing.parse(args.parse_system, chunks, sid, iid), guard_hits))
    responses = _run_qa(_qa_questions(), records, args.system, guard=guard)
    for qid, resp in responses.items():
        write_json(os.path.join(args.out, qid + ".json"), resp)
    print(f"[qa-e2e] parse={args.parse_system} qa={args.system} "
          f"parsed={len(records)} answered={len(responses)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parse", "qa", "qa-e2e"])
    ap.add_argument("--system", default="tool_assisted")
    ap.add_argument("--parse-system", default="tool_assisted")
    ap.add_argument("--set", default="parsing", choices=["parsing", "qa"])
    ap.add_argument("--records", default=os.path.join(DATA, "parsed_docs", "qa"),
                    help="parsed IncidentRecords to answer over (default: gold parsed_docs/qa)")
    ap.add_argument("--out", default=repo_path("out", "preds"))
    ap.add_argument("--no-guard", action="store_true",
                    help="disable guardrail sanitization (for before/after safety comparison)")
    ap.add_argument("--mock", action="store_true",
                    help="offline: stub the LLM (no model endpoint needed)")
    args = ap.parse_args()

    if args.mock:
        _apply_mock()

    os.makedirs(args.out, exist_ok=True)

    if args.mode == "parse":
        cmd_parse(args)
    elif args.mode == "qa":
        cmd_qa(args)
    else:
        cmd_qa_e2e(args)


if __name__ == "__main__":
    main()
