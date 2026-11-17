import argparse
import csv
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import (  # noqa: E402
    _PARSING_FIELDS, _evidence_for, _get_path, _is_known, agreement,
    judge_parsing_record, judge_qa_answer,
)
from utils import gold_records, load_json_by_name, questions, write_json  # noqa: E402

# Excel in a Russian locale splits CSV on ';'.
DELIM = ";"
_YES = {"correct", "true", "1", "yes", "да", "+"}
_NO = {"incorrect", "false", "0", "no", "нет", "-"}


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        print("nothing to export")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=DELIM)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def _as_bool(value):
    v = str(value or "").strip().lower()
    return True if v in _YES else False if v in _NO else None


def cmd_parsing(args) -> None:
    gold, preds = gold_records("parsing"), load_json_by_name(args.pred_dir)
    for sid in sorted(gold):
        out_path = os.path.join(args.out, sid + ".json")
        if sid not in preds:
            print("no prediction", sid)
            continue
        if os.path.exists(out_path) and not args.force:
            continue
        v = judge_parsing_record(gold[sid], preds[sid])
        write_json(out_path, v)
        print(sid, "judge_missing", len(v["judge_missing"]))


def cmd_qa(args) -> None:
    rows, preds = questions(), load_json_by_name(args.pred_dir)
    for qid, row in rows.items():
        out_path = os.path.join(args.out, qid + ".json")
        if qid not in preds:
            print("no response", qid)
            continue
        if os.path.exists(out_path) and not args.force:
            continue
        v = judge_qa_answer(row["question"], row, preds[qid])
        write_json(out_path, v)
        print(qid, v["judged_by"], v["answer_correct"])


def cmd_export_parsing(args) -> None:
    gold, preds, verdicts = gold_records("parsing"), load_json_by_name(args.pred_dir), load_json_by_name(args.judge_dir)
    rows = []
    for sid, v in verdicts.items():
        if sid not in gold or sid not in preds:
            continue
        for path in _PARSING_FIELDS:
            g, p = _get_path(gold[sid], path), _get_path(preds[sid], path)
            if not (_is_known(g) and _is_known(p)):
                continue
            rows.append({"source_file_id": sid, "path": path,
                         "gold": json.dumps(g, ensure_ascii=False), "pred": json.dumps(p, ensure_ascii=False),
                         "pred_evidence": " | ".join(_evidence_for(preds[sid], path)),
                         "judge_value": v["field_verdicts_value"][path], "judge_strict": v["field_verdicts"][path],
                         "human_value": "", "human_strict": ""})
    random.Random(0).shuffle(rows)
    if args.sample:
        rows = rows[: args.sample]
    _write_csv(os.path.join(args.judge_dir, "human.csv"), rows)


def cmd_export_qa(args) -> None:
    qrows, preds, verdicts = questions(), load_json_by_name(args.pred_dir), load_json_by_name(args.judge_dir)
    rows = []
    for qid, v in sorted(verdicts.items()):
        if v.get("judged_by") != "llm" or qid not in preds:
            continue
        cited = " | ".join(((c.get("evidence") or {}).get("chunk") or {}).get("text", "")
                           for c in preds[qid].get("citations") or [])
        rows.append({"question_id": qid, "question": qrows[qid]["question"],
                     "reference": qrows[qid].get("reference_answer", ""),
                     "predicted": preds[qid].get("answer", ""), "cited": cited,
                     "judge_answer_correct": v.get("answer_correct"), "human_answer_correct": ""})
    _write_csv(os.path.join(args.judge_dir, "human.csv"), rows)


def cmd_agreement(args) -> None:
    with open(args.human, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=DELIM))
    pairs = [("judge_strict", "human_strict"), ("judge_value", "human_value"),
             ("judge_answer_correct", "human_answer_correct")]
    for judge_col, human_col in pairs:
        if not rows or judge_col not in rows[0]:
            continue
        j, h = [], []
        for r in rows:
            hv, jv = _as_bool(r.get(human_col)), _as_bool(r.get(judge_col))
            if hv is not None and jv is not None:
                j.append(jv)
                h.append(hv)
        print(f"{judge_col} vs {human_col}: {agreement(j, h)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parsing", "qa", "export-parsing", "export-qa", "agreement"])
    ap.add_argument("--pred-dir")
    ap.add_argument("--judge-dir")
    ap.add_argument("--out")
    ap.add_argument("--human")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"parsing": cmd_parsing, "qa": cmd_qa, "export-parsing": cmd_export_parsing,
     "export-qa": cmd_export_qa, "agreement": cmd_agreement}[args.mode](args)


if __name__ == "__main__":
    main()
