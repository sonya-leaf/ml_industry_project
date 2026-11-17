import json

from llm import DEFAULT_MODEL, complete
from qa.baseline import build_response
from qa.retrieve import records_context
from qa.prompts import REFORMULATE_SYSTEM, REFORMULATE_VERSION, TOOL_ASSISTED_SYSTEM, TOOL_ASSISTED_SYSTEM_VERSION, qa_user_prompt
from utils import extract_json_object
from tools import abbreviation_expansions, get_document_chunks, query_incidents

MODEL = DEFAULT_MODEL


def _expand(question: str) -> str:
    return "\n".join(f"- {tok}: {full}" for tok, full in abbreviation_expansions(question))


def _parse_queries(raw: str) -> list[str]:
    obj = extract_json_object(raw)
    if isinstance(obj, list):
        return [str(q) for q in obj if str(q).strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


_FACT_KEYS = ("occurred_at", "event_type", "report_stage", "fault", "observations")


def _facts(rec: dict) -> str:
    facts = {k: rec[k] for k in _FACT_KEYS if rec.get(k) not in (None, [], "")}
    return json.dumps(facts, ensure_ascii=False)


def _evidence_context(records_by_id: dict, ids: list[str], chunk_store: dict) -> str:
    parts = []
    for iid in ids:
        rec = records_by_id.get(iid)
        if not rec:
            continue
        paths_by_text: dict[str, list[str]] = {}
        for ev in rec.get("evidence") or []:
            text = (ev.get("chunk") or {}).get("text")
            if text:
                paths = paths_by_text.setdefault(text, [])
                paths.extend(p for p in ev.get("field_paths") or [] if p not in paths)
        lines = [f"[{iid}]", "Facts (not a citation):", _facts(rec),
                 "Evidence objects (copy one verbatim into citations):"]
        objects = []
        for chunk in get_document_chunks(iid, chunk_store):
            objects.append(json.dumps({"field_paths": paths_by_text.get(chunk.get("text"), []),
                                       "chunk": chunk}, ensure_ascii=False))
        lines.extend(objects or ["(no evidence chunks)"])
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def answer(question: str, index, records_by_id: dict, top_k: int = 5, source_id: str = "qa") -> dict:
    expansions = _expand(question)
    preprompt = f"Question: {question}\nAbbreviation expansions:\n{expansions or 'none'}"

    raw_qs = complete(REFORMULATE_SYSTEM, preprompt, source_id=source_id,
                      system_version=REFORMULATE_VERSION, model=MODEL)
    # The original question is always searched. Reformulations are extra probes, and they
    # are searched deeper than top_k so a hit just outside one query's first page survives
    # the merge. query_incidents keeps each incident's best score.
    probes: list[str] = []
    for q in (question, *_parse_queries(raw_qs)):
        q = (q or "").strip()
        if q and q not in probes:
            probes.append(q)
    ranked = query_incidents(probes[:4], top_k=top_k, index=index, pool=max(top_k * 4, 20))
    ids = [doc_id for doc_id, _ in ranked]

    ctx = _evidence_context(records_by_id, ids, getattr(index, "chunk_store", {}))
    if not ctx.strip():
        # No evidence chunks available (e.g. empty chunk_store) — still give the model the
        # retrieved records so it never answers from an empty prompt.
        ctx = records_context(records_by_id, ids)
    raw = complete(TOOL_ASSISTED_SYSTEM, qa_user_prompt(question, ctx), source_id=source_id,
                   system_version=TOOL_ASSISTED_SYSTEM_VERSION, model=MODEL)
    return build_response(raw)
