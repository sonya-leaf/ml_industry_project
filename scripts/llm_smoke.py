import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guardrails import WARNING, sanitize_chunks
from llm import DEFAULT_MODEL, complete
from parsing.direct import build_record_from_output
from parsing.extract import DocumentChunk, extract_document
from parsing.prompts import DIRECT_SYSTEM, DIRECT_SYSTEM_VERSION, parse_user_prompt
from utils import repo_path, validate, write_json

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/llm_smoke.py <path to one .docx>")

path = sys.argv[1]
sid = os.path.basename(path)
iid = "TESTDOC-" + os.path.splitext(sid)[0]

chunks = extract_document(path)
clean, guard_hits = sanitize_chunks([c.to_dict() for c in chunks])
chunks = [DocumentChunk.from_dict(d) for d in clean]

raw = complete(
    DIRECT_SYSTEM,
    parse_user_prompt(sid, chunks),
    source_id=sid,
    trace_id=iid,
    system_version=DIRECT_SYSTEM_VERSION,
    model=DEFAULT_MODEL,
)

record = build_record_from_output(raw, sid, iid)
if guard_hits and WARNING not in record.get("warnings", []):
    record.setdefault("warnings", []).append(WARNING)

ok, errors = validate(record, "incident_record")
out_path = repo_path("out", "test_document_direct_call.json")
write_json(out_path, record)

print("reply_chars", len(raw))
print("reply_ends_with_brace", raw.rstrip().endswith("}"))
print("schema_ok", ok)
print("schema_errors", errors[:8])
print("record_warnings", record.get("warnings"))
print("observations", len(record.get("observations") or []),
      "evidence", len(record.get("evidence") or []))
print("wrote", out_path)
