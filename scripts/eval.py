import argparse
import os
import sys

# bootstrap repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (  # noqa: E402
    citation_accuracy,
    citation_grounding,
    field_f1_from_judge,
    field_f1_metrics,
    injection_refusal_rate,
    parse_ids_list,
    recall_at_k,
    refusal_on_missing_evidence,
    schema_valid_rate,
    status_accuracy,
)
from guardrails import detect_injection, sanitize_chunks, sanitize_question  # noqa: E402
import parsing  # noqa: E402
from parsing.extract import extract_document  # noqa: E402
from qa import retrieve  # noqa: E402
from utils import gold_records, load_json_by_name, load_records, questions, raw_docs, repo_path  # noqa: E402


# ---- parsing ---------------------------------------------------------------


def cmd_parsing(args) -> None:
    gold = gold_records(args.set)
    if args.system == "baseline":
        preds = {}
        for path in raw_docs(args.set):
            sid = os.path.basename(path)  # source_file_id includes the extension, e.g. "02.docx"
            chunks = extract_document(path)
            preds[sid] = parsing.parse("baseline", chunks, sid, f"PARSING-{sid}")
    else:
        if not args.pred_dir:
            sys.exit("LLM systems need --pred-dir with model outputs (generate them via scripts/run.py)")
        preds = load_json_by_name(args.pred_dir)

    gold_list = [gold[k] for k in sorted(gold) if k in preds]
    pred_list = [preds[k] for k in sorted(gold) if k in preds]
    per_field, macro = field_f1_metrics(gold_list, pred_list)
    rate, n = schema_valid_rate(pred_list, "incident_record")
    print(f"parsing [{args.system}] set={args.set}  n={len(gold_list)}")
    print(f"  schema-valid rate: {rate:.3f} ({n}/{len(gold_list)})")
    print(f"  field macro-F1:    {macro:.3f}")
    for path, f in per_field.items():
        print(f"    {path:38s} P={f['precision']:.2f} R={f['recall']:.2f} F1={f['f1']:.2f}")

# ---- retrieval (no LLM) ----------------------------------------------------


def cmd_retrieval(args) -> None:
    from qa.retrieve import HybridRetriever
    gold = gold_records("qa")
    records = list(gold.values())
    doc_index, _ = retrieve.build_incident_corpus(records)
    hybrid = HybridRetriever(records, expand=True, stem=True)
    hybrid_noexp = HybridRetriever(records, expand=False, stem=True)
    hybrid_nostem = HybridRetriever(records, expand=True, stem=False)
    rows = list(questions().values())
    k = int(args.k)
    pairs = [(r["question"], parse_ids_list(r.get("relevant_incident_ids"))) for r in rows]
    pairs = [(q, g) for q, g in pairs if g]
    n = len(pairs)

    def recall(searcher):
        total = 0.0
        for q, g in pairs:
            total += recall_at_k([d for d, _ in searcher(q, top_k=k)], g, k=k)
        return total / n

    print(f"retrieval Recall@{k}  (n={n})")
    print(f"  doc BM25 (original query)        {recall(lambda q, top_k=k: doc_index.search(q, top_k=top_k)):.3f}")
    print(f"  hybrid full (ref+facility+chunk)  {recall(hybrid.search):.3f}")
    print(f"  hybrid w/o abbr expansion        {recall(hybrid_noexp.search):.3f}")
    print(f"  hybrid w/o stemming (no facility){recall(hybrid_nostem.search):.3f}")

# ---- QA gold ---------------------------------------------------------------


def cmd_qa_gold(args) -> None:
    rows = list(questions().values())
    if not args.pred_dir:
        sys.exit("QA needs --pred-dir with QAResponse files (generate them via scripts/run.py)")
    resp = load_json_by_name(args.pred_dir)
    gold_rows = [r for r in rows if r["question_id"] in resp]
    preds = [resp[r["question_id"]] for r in gold_rows]
    records_dir = args.records or repo_path("data", "golden_set", "parsed_docs", "qa")
    records = load_records(records_dir)
    records_by_id = {r["incident_id"]: r for r in records}
    print(f"qa-gold [{args.system}]  n={len(preds)}  grounding records={records_dir}")
    print(f"  status accuracy:           {status_accuracy(preds, gold_rows):.3f}")
    print(f"  citation accuracy (proxy): {citation_accuracy(preds, gold_rows):.3f}")
    print(f"  citation grounding:        {citation_grounding(preds, records_by_id):.3f}")
    print(f"  refusal on missing:        {refusal_on_missing_evidence(preds, gold_rows):.3f}")
    print(f"  injection refusal:         {injection_refusal_rate(preds, gold_rows):.3f}")

# ---- judge-based metrics (verdicts from the LLM judge via scripts/judge.py) ---


def cmd_judge_parsing(args) -> None:
    gold = gold_records("parsing")
    preds = load_json_by_name(args.pred_dir)
    jv = {d["source_file_id"]: d for d in load_json_by_name(args.judge_dir).values()}
    keys = [k for k in sorted(gold) if k in preds and k in jv]
    gold_list = [gold[k] for k in keys]
    pred_list = [preds[k] for k in keys]
    _, exact = field_f1_metrics(gold_list, pred_list)
    _, strict = field_f1_from_judge([jv[k]["field_verdicts"] for k in keys], gold_list, pred_list)
    _, value = field_f1_from_judge([jv[k]["field_verdicts_value"] for k in keys], gold_list, pred_list)
    missing = sum(len(jv[k].get("judge_missing", [])) for k in keys)
    agree_exact = _agreement(gold_list, pred_list, [jv[k]["field_verdicts_value"] for k in keys])
    print(f"judge-parsing [{args.system}]  n={len(keys)}")
    print(f"  field macro-F1 (judge, value + evidence): {strict:.3f}")
    print(f"  field macro-F1 (judge, value only):       {value:.3f}")
    print(f"  field macro-F1 (exact match):             {exact:.3f}")
    print(f"  fields the judge never answered:          {missing}")
    print(f"  judge agrees with exact match (diagnostic, not human agreement): {agree_exact:.3f}")


def cmd_judge_qa(args) -> None:
    rows = list(questions().values())
    resp = load_json_by_name(args.pred_dir)
    jv = load_json_by_name(args.judge_dir)
    acc = [jv[r["question_id"]] for r in rows if r["question_id"] in resp and r["question_id"] in jv]
    n = len(acc)
    failed = sum(1 for a in acc if a.get("answer_correct") is None)
    print(f"judge-qa [{args.system}]  n={n}")
    print(f"  status accuracy:                   {sum(a.get('status_correct') is True for a in acc) / n if n else 0:.3f}")
    print(f"  evidence-grounded answer accuracy: {sum(a.get('answer_correct') is True for a in acc) / n if n else 0:.3f}")
    print(f"  judge failures (counted wrong):    {failed}")


def _agreement(gold_list, pred_list, vd) -> float:
    from evaluation import _PARSING_FIELDS, _get_path, _is_known, _values_equal
    agree = tot = 0
    for g, p, j in zip(gold_list, pred_list, vd):
        for path in _PARSING_FIELDS:
            gv = _get_path(g, path)
            if not _is_known(gv):
                continue
            exact = _is_known(_get_path(p, path)) and _values_equal(gv, _get_path(p, path))
            agree += (exact == (j.get(path) == "correct"))
            tot += 1
    return agree / tot if tot else 1.0

# ---- safety (guardrail detection) ------------------------------------------


def cmd_safety(args) -> None:
    # before (raw) vs after (sanitized) at the guardrail input level.
    rows = list(questions().values())
    inj_q = [r for r in rows if str(r.get("question_contains_prompt_injection") or "").strip().lower() in {"true", "1"}]
    det_q = sum(1 for r in inj_q if detect_injection(r["question"]))
    clean_q = sum(1 for r in inj_q if not detect_injection(sanitize_question(r["question"])[0]))
    print(f"guardrail questions: detected {det_q}/{len(inj_q)}; fully sanitized {clean_q}/{len(inj_q)}")

    inj_docs = [os.path.basename(p) for p in raw_docs("parsing") if "_inj" in os.path.basename(p)]
    det_d = clean_d = 0
    for name in inj_docs:
        chunks = [c.to_dict() for c in extract_document(repo_path("data", "golden_set", "raw_docs", "parsing", name))]
        if any(detect_injection(c.get("text", "")) for c in chunks):
            det_d += 1
        clean, _ = sanitize_chunks(chunks)
        if not any(detect_injection(c.get("text", "")) for c in clean):
            clean_d += 1
    print(f"guardrail documents: detected {det_d}/{len(inj_docs)}; fully sanitized {clean_d}/{len(inj_docs)}")

    if args.pred_dir:
        resp = load_json_by_name(args.pred_dir)
        gold = [r for r in rows if r["question_id"] in resp]
        preds = [resp[r["question_id"]] for r in gold]
        records = load_records(repo_path("data", "golden_set", "parsed_docs", "qa"))
        records_by_id = {r["incident_id"]: r for r in records}
        print(f"injection refusal on QA preds: {injection_refusal_rate(preds, gold):.3f}")
        print(f"citation grounding on QA preds: {citation_grounding(preds, records_by_id):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Golden-set evaluation")
    ap.add_argument("mode", choices=["parsing", "qa-gold", "retrieval", "safety",
                                     "judge-parsing", "judge-qa"])
    ap.add_argument("--system", default="baseline")
    ap.add_argument("--set", default="parsing", choices=["parsing", "qa"])
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--records", default=None,
                    help="records the QA run answered over, for citation grounding (default: gold qa)")
    ap.add_argument("--judge-dir", default=None)
    ap.add_argument("--k", default="5")
    args = ap.parse_args()

    if args.mode == "parsing":
        cmd_parsing(args)
    if args.mode == "retrieval":
        cmd_retrieval(args)
    if args.mode == "qa-gold":
        cmd_qa_gold(args)
    if args.mode == "safety":
        cmd_safety(args)
    if args.mode == "judge-parsing":
        if not args.judge_dir:
            sys.exit("judge-parsing needs --pred-dir and --judge-dir")
        cmd_judge_parsing(args)
    if args.mode == "judge-qa":
        if not args.judge_dir:
            sys.exit("judge-qa needs --pred-dir and --judge-dir")
        cmd_judge_qa(args)


if __name__ == "__main__":
    main()
