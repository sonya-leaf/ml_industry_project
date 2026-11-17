import argparse
import csv
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # after the root: scripts/qa.py must not shadow qa/

import parsing  # noqa: E402
import qa  # noqa: E402
from evaluation import (  # noqa: E402
    citation_accuracy,
    citation_grounding,
    field_f1_from_judge,
    field_f1_metrics,
    injection_refusal_rate,
    judge_parsing_record,
    judge_qa_answer,
    parse_ids_list,
    recall_at_k,
    refusal_on_missing_evidence,
    schema_valid_rate,
    status_accuracy,
)
from guardrails import add_warning, sanitize_question  # noqa: E402
from parsing.extract import extract_document, prepare_chunks  # noqa: E402
from qa.retrieve import index_records  # noqa: E402
from run import _incident_id  # noqa: E402
from utils import gold_records, questions, raw_docs, read_json, repo_path, write_json  # noqa: E402

SYSTEMS = ["baseline", "direct", "tool_assisted"]
_FAILED_QA = {"schema_version": "1.0", "status": "insufficient_evidence", "answer": "", "citations": []}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cached(path: str, reuse: bool, compute):
    if reuse and os.path.exists(path):
        return read_json(path), True
    try:
        obj = compute()
    except Exception as exc:  # one failing item must not abort the whole comparison
        _log(f"  FAILED {os.path.basename(path)}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=1)
        return None, False
    write_json(path, obj)
    return obj, True


class _RecordingIndex:
    def __init__(self, index):
        self._index = index
        self.calls: list[list[tuple[str, float]]] = []

    def search(self, query, top_k=5, **kw):
        res = self._index.search(query, top_k=top_k, **kw)
        self.calls.append(res)
        return res

    def ranked(self, top_k: int) -> list[str]:
        # Same merge as tools.query_incidents: best score across queries, not the sum.
        # A sum lets a weak match on every reformulation outrank one precise hit.
        merged: dict[str, float] = {}
        for res in self.calls:
            for doc_id, score in res:
                prev = merged.get(doc_id)
                if prev is None or score > prev:
                    merged[doc_id] = score
        return [d for d, _ in sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]

    def __getattr__(self, name):
        return getattr(self._index, name)


# ---- parsing ---------------------------------------------------------------


def parse_set(system: str, set_name: str, out_dir: str, reuse: bool, guard: bool) -> tuple[dict, int]:
    preds, failed = {}, 0
    for path in raw_docs(set_name):
        sid = os.path.basename(path)

        def compute(path=path, sid=sid):
            chunks, hits = prepare_chunks(extract_document(path), guard)
            return add_warning(parsing.parse(system, chunks, sid, _incident_id(sid, set_name)), hits)

        rec, ok = _cached(os.path.join(out_dir, sid + ".json"), reuse, compute)
        if ok:
            preds[sid] = rec
        else:
            failed += 1
    _log(f"  parse {set_name}: {len(preds)} ok, {failed} failed")
    return preds, failed


def score_parsing(gold: dict, preds: dict, prefix: str, system: str, field_rows: list) -> dict:
    keys = sorted(gold)
    gold_list = [gold[k] for k in keys]
    pred_list = [preds.get(k, {}) for k in keys]
    per_field, macro = field_f1_metrics(gold_list, pred_list)
    rate, _ = schema_valid_rate(pred_list, "incident_record")
    for path, f in per_field.items():
        field_rows.append({"system": system, "set": prefix, "field": path,
                           "precision": f["precision"], "recall": f["recall"], "f1": f["f1"]})
    return {f"{prefix}_n": len(keys), f"{prefix}_schema_valid": round(rate, 3),
            f"{prefix}_field_f1_exact": macro}


def judge_parsing(gold: dict, preds: dict, out_dir: str, reuse: bool) -> dict:
    keys = sorted(gold)
    verdicts = []
    for sid in keys:
        v, ok = _cached(os.path.join(out_dir, sid + ".json"), reuse,
                        lambda sid=sid: judge_parsing_record(gold[sid], preds.get(sid, {})))
        verdicts.append(v if ok else {"field_verdicts": {}, "field_verdicts_value": {}})
    gold_list = [gold[k] for k in keys]
    pred_list = [preds.get(k, {}) for k in keys]
    _, strict = field_f1_from_judge([v["field_verdicts"] for v in verdicts], gold_list, pred_list)
    _, value = field_f1_from_judge([v["field_verdicts_value"] for v in verdicts], gold_list, pred_list)
    return {"parsing_field_f1_judge_strict": strict, "parsing_field_f1_judge_value": value}

# ---- QA --------------------------------------------------------------------


def run_qa(system: str, records: list[dict], sys_dir: str, reuse: bool, guard: bool, k: int) -> tuple[dict, dict, int]:
    index, records_by_id = index_records(records)
    retrieved_path = os.path.join(sys_dir, "qa_retrieved.json")
    retrieved = read_json(retrieved_path) if reuse and os.path.exists(retrieved_path) else {}
    responses, failed = {}, 0
    for qid, row in questions().items():
        question = sanitize_question(row["question"])[0] if guard else row["question"]
        rec_index = _RecordingIndex(index)

        def compute(question=question, qid=qid, rec_index=rec_index):
            return qa.answer(system, question, rec_index, records_by_id, top_k=k, source_id=qid)

        resp, ok = _cached(os.path.join(sys_dir, "qa", qid + ".json"), reuse, compute)
        if rec_index.calls:
            retrieved[qid] = rec_index.ranked(k)
        responses[qid] = resp if ok else dict(_FAILED_QA)
        failed += not ok
    write_json(retrieved_path, retrieved)
    _log(f"  qa: {len(responses) - failed} ok, {failed} failed")
    return responses, retrieved, failed


def score_qa(responses: dict, retrieved: dict, records: list[dict], k: int) -> dict:
    rows = list(questions().values())
    preds = [responses[r["question_id"]] for r in rows]
    records_by_id = {r["incident_id"]: r for r in records}
    rate, _ = schema_valid_rate(preds, "qa_response")
    pairs = [(retrieved.get(r["question_id"], []), parse_ids_list(r.get("relevant_incident_ids"))) for r in rows]
    pairs = [(p, g) for p, g in pairs if g]
    recall = sum(recall_at_k(p, g, k=k) for p, g in pairs) / len(pairs) if pairs else 0.0
    return {
        "qa_n": len(preds),
        "qa_schema_valid": round(rate, 3),
        f"qa_recall@{k}": round(recall, 3),
        "qa_status_accuracy": round(status_accuracy(preds, rows), 3),
        "qa_citation_accuracy": round(citation_accuracy(preds, rows), 3),
        "qa_citation_grounding": round(citation_grounding(preds, records_by_id), 3),
        "qa_refusal_on_missing": round(refusal_on_missing_evidence(preds, rows), 3),
        "qa_injection_refusal": round(injection_refusal_rate(preds, rows), 3),
    }


def judge_qa(responses: dict, out_dir: str, reuse: bool) -> dict:
    verdicts = []
    for qid, row in questions().items():
        v, ok = _cached(os.path.join(out_dir, qid + ".json"), reuse,
                        lambda qid=qid, row=row: judge_qa_answer(row["question"], row, responses[qid]))
        verdicts.append(v if ok else {"answer_correct": None})
    n = len(verdicts)
    return {"qa_answer_accuracy_judge": round(sum(v.get("answer_correct") is True for v in verdicts) / n, 3) if n else 0.0,
            "qa_judge_failures": sum(v.get("answer_correct") is None for v in verdicts)}

# ---- driver ----------------------------------------------------------------


def run_system(system: str, args, field_rows: list) -> dict:
    sys_dir = os.path.join(args.out, system)
    guard = not args.no_guard
    row: dict = {"system": system}

    _log(f"[{system}] parsing set")
    gold_p = gold_records("parsing")
    preds_p, fail_p = parse_set(system, "parsing", os.path.join(sys_dir, "parsing"), args.reuse, guard)
    row.update(score_parsing(gold_p, preds_p, "parsing", system, field_rows), parsing_failures=fail_p)
    if args.judge:
        row.update(judge_parsing(gold_p, preds_p, os.path.join(sys_dir, "judge_parsing"), args.reuse))

    _log(f"[{system}] qa set: parse raw docs")
    gold_q = gold_records("qa")
    preds_q, fail_q = parse_set(system, "qa", os.path.join(sys_dir, "qa_records"), args.reuse, guard)
    row.update(score_parsing(gold_q, preds_q, "qa_records", system, field_rows), qa_records_failures=fail_q)

    _log(f"[{system}] qa set: answer questions")
    records = [preds_q[k] for k in sorted(preds_q)]
    responses, retrieved, fail_a = run_qa(system, records, sys_dir, args.reuse, guard, args.k)
    row.update(score_qa(responses, retrieved, records, args.k), qa_failures=fail_a)
    if args.judge:
        row.update(judge_qa(responses, os.path.join(sys_dir, "judge_qa"), args.reuse))
    return row


def _write_table(path: str, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _print_table(rows: list[dict]) -> None:
    metrics = [c for c in rows[0] if c != "system"]
    width = max(len(m) for m in metrics)
    print("\n" + "metric".ljust(width) + "".join(r["system"].rjust(15) for r in rows))
    for m in metrics:
        print(m.ljust(width) + "".join(str(r.get(m, "")).rjust(15) for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run parsing + QA for all systems and score against the golden set")
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--out", default=repo_path("out", "compare", time.strftime("%Y%m%d_%H%M")))
    ap.add_argument("--reuse", action="store_true", help="keep existing predictions/verdicts in --out")
    ap.add_argument("--judge", action="store_true", help="also run the LLM judge (Ollama Mistral)")
    ap.add_argument("--no-guard", action="store_true", help="disable guardrail sanitization")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _log(f"out={args.out} systems={args.systems} judge={args.judge} guard={not args.no_guard}")
    rows, field_rows = [], []
    for system in args.systems:
        rows.append(run_system(system, args, field_rows))
        _write_table(os.path.join(args.out, "results.csv"), rows)
        _write_table(os.path.join(args.out, "parsing_fields.csv"), field_rows)
    _print_table(rows)
    print(f"\nresults: {os.path.join(args.out, 'results.csv')}")


if __name__ == "__main__":
    main()
