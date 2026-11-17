from llm import DEFAULT_MODEL, complete
from qa.prompts import DIRECT_SYSTEM, DIRECT_SYSTEM_VERSION, qa_user_prompt
from qa.retrieve import records_context
from utils import extract_json_object

MODEL = DEFAULT_MODEL

_EMPTY = {"schema_version": "1.0", "status": "insufficient_evidence", "answer": "", "citations": []}


def build_response(raw: str) -> dict:
    obj = extract_json_object(raw)
    if obj is None:
        return dict(_EMPTY)
    return _coerce_response(obj)


def _coerce_response(obj: dict) -> dict:
    status = str(obj.get("status") or "").strip().lower()
    answer = obj.get("answer") if isinstance(obj.get("answer"), str) else ""
    citations = [c for c in obj.get("citations", [])
                 if isinstance(c, dict) and isinstance(c.get("incident_id"), str) and c.get("incident_id")]
    if status == "insufficient_evidence" or not answer.strip():
        return {"schema_version": "1.0", "status": "insufficient_evidence", "answer": answer, "citations": []}
    return {"schema_version": "1.0", "status": "answered", "answer": answer, "citations": citations}


def answer(question: str, index, records_by_id: dict, top_k: int = 5, source_id: str = "qa") -> dict:
    ids = [doc_id for doc_id, _ in index.search(question, top_k=top_k)]
    ctx = records_context(records_by_id, ids)
    raw = complete(DIRECT_SYSTEM, qa_user_prompt(question, ctx), source_id=source_id,
                   system_version=DIRECT_SYSTEM_VERSION, model=MODEL)
    return build_response(raw)
