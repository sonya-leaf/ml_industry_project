import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import parse_ids_list, recall_at_k
from guardrails import sanitize_question
from llm import DEFAULT_MODEL, complete
from qa.baseline import build_response
from qa.prompts import (
    DIRECT_SYSTEM,
    DIRECT_SYSTEM_VERSION,
    REWRITE_SYSTEM,
    REWRITE_VERSION,
    qa_user_prompt,
)
from qa.retrieve import HybridRetriever, records_context
from utils import extract_json_object, read_csv_rows, read_json, repo_path, validate, write_json

ap = argparse.ArgumentParser()
ap.add_argument("records", help="one IncidentRecord .json file or a directory of them")
ap.add_argument("--question", help="free-form question")
ap.add_argument("--question-id", help="take the question and gold ids from questions/qa.csv")
ap.add_argument("--system", choices=["baseline", "direct"], default="direct")
ap.add_argument("--top-k", type=int, default=5)
ap.add_argument("--no-expand", action="store_true", help="plain BM25, no abbreviation expansion")
args = ap.parse_args()

if os.path.isfile(args.records):
    paths = [args.records]
else:
    paths = sorted(glob.glob(os.path.join(args.records, "*.json")))

records, n_invalid = [], 0
for p in paths:
    rec = read_json(p)
    if not validate(rec, "incident_record")[0]:
        n_invalid += 1
    if rec.get("incident_id"):
        records.append(rec)
if not records:
    sys.exit("no records with incident_id")
records_by_id = {r["incident_id"]: r for r in records}
if len(records_by_id) != len(records):
    sys.exit("duplicate incident_id in records; retrieval would merge incidents")

gold_ids: list[str] = []
qid = "adhoc"
if args.question_id:
    rows = read_csv_rows(repo_path("data", "golden_set", "questions", "qa.csv"))
    row = next((r for r in rows if r["question_id"] == args.question_id), None)
    if row is None:
        sys.exit(f"question_id {args.question_id} not in qa.csv")
    question, qid = row["question"], args.question_id
    gold_ids = parse_ids_list(row.get("relevant_incident_ids"))
elif args.question:
    question = args.question
else:
    sys.exit("give --question or --question-id")

clean_q, removed = sanitize_question(question)

query = clean_q
if args.system == "direct":
    query = complete(REWRITE_SYSTEM, f"Question: {clean_q}", source_id=qid,
                     system_version=REWRITE_VERSION, model=DEFAULT_MODEL).strip() or clean_q

index = HybridRetriever(records, expand=not args.no_expand)
ids = [iid for iid, _ in index.search(query, top_k=args.top_k)]

raw = complete(DIRECT_SYSTEM, qa_user_prompt(clean_q, records_context(records_by_id, ids)),
               source_id=qid, system_version=DIRECT_SYSTEM_VERSION, model=DEFAULT_MODEL)
resp = build_response(raw)
ok, errors = validate(resp, "qa_response")

citations = resp.get("citations", [])
outside = [c.get("incident_id") for c in citations if c.get("incident_id") not in ids]
verbatim = 0
for c in citations:
    rec = records_by_id.get(c.get("incident_id"))
    text = ((c.get("evidence") or {}).get("chunk") or {}).get("text", "")
    if rec and text and any(text in (e.get("chunk") or {}).get("text", "") for e in rec.get("evidence", [])):
        verbatim += 1

out_path = repo_path("out", f"test_qa_{qid}.json")
write_json(out_path, {"question_id": qid, "system": args.system, "query": query,
                      "retrieved": ids, "response": resp})

print("records", len(records), "schema_invalid_records", n_invalid)
print("guardrail_removed_spans", len(removed))
print("retrieved", ids)
if gold_ids:
    print("gold", gold_ids, f"recall@{args.top_k}", round(recall_at_k(ids, gold_ids, args.top_k), 3))
print("reply_has_json", extract_json_object(raw) is not None)
print("status", resp.get("status"))
print("schema_ok", ok, errors[:5])
print("citations", len(citations), "verbatim_in_records", verbatim, "cited_outside_retrieved", outside)
print("wrote", out_path)
