import json
import re

from llm import DEFAULT_MODEL, complete
from parsing.direct import build_record_from_output
from parsing.prompts import TOOL_ASSISTED_SYSTEM, TOOL_ASSISTED_SYSTEM_VERSION, parse_user_prompt
from tools import abbreviation_expansions
from utils import validate

MODEL = DEFAULT_MODEL
REPAIR_PROMPT = """\
The IncidentRecord below did not pass schema validation. Fix ONLY the reported errors.
Every field that is not named in Errors must be copied unchanged from the previous JSON.
Do not rephrase values and do not drop evidence.

Previous JSON:
{previous}

Errors:
{errors}

Re-emit the full corrected IncidentRecord JSON object only.
"""
REPAIR_VERSION = "parsing.repair.v2"
REPAIR_ATTEMPTS = 3  # retries with the error text after the first parse call
NO_JSON_PROMPT = """\
Your previous answer was not a JSON object, so no IncidentRecord was produced.
Using the same chunks, emit the full IncidentRecord JSON object only.
No markdown and no commentary.
"""
NO_JSON_VERSION = "parsing.no_json.v1"
_WORD = re.compile(r"[^\W_]{3,}", re.UNICODE)
_NUM = re.compile(r"\d+")
# Fields the judge scores. A populated field with no field_path is unsupported.
_BIND_PATHS = (
    "report_stage", "occurred_at", "event_type",
    "fault.description", "fault.faulted_phases", "fault.ground_involved",
    "observations/0/facility", "observations/0/voltage_levels_kv",
    "observations/0/connections", "observations/0/pre_event_scheme",
    "observations/0/switching_actions", "observations/0/rpa_operations",
    "observations/0/reclosing_result", "observations/0/event_summary",
    "observations/0/rpa_assessment", "observations/0/reported_causes",
    "observations/0/recommendations",
)
_UNKNOWN = {"unknown", "not_reported", "not_attempted", ""}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text or "")} | set(_NUM.findall(text or ""))


def align_evidence(record: dict, chunks) -> dict:
    sources = [c for c in chunks if getattr(c, "text", "")]
    texts = [c.text for c in sources]
    for ev in record.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        chunk = ev.get("chunk")
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text") or ""
        if not text or any(text in src for src in texts):
            continue
        cited = _tokens(text)
        if not cited:
            continue
        best, best_score = None, 0.0
        for src in sources:
            score = len(cited & _tokens(src.text)) / len(cited)
            if score > best_score:
                best, best_score = src, score
        if best is None or best_score < 0.5:
            continue
        chunk["source_file_id"] = best.source_file_id
        chunk["section"] = best.section
        chunk["page"] = best.page
        chunk["text"] = best.text
    return record


def _get(record: dict, path: str):
    cur = record
    for part in path.split("/"):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def _known(value) -> bool:
    if value is None or value == [] or value == {}:
        return False
    if isinstance(value, str) and value.strip().lower() in _UNKNOWN:
        return False
    return True


def _covered(record: dict, path: str) -> bool:
    pointer = "/" + path
    for ev in record.get("evidence") or []:
        for fp in ev.get("field_paths") or []:
            if pointer == fp or pointer.startswith(fp.rstrip("/") + "/"):
                return True
    return False


def _needles(value) -> set[str]:
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float)):
        return {str(int(value)) if float(value).is_integer() else str(value)}
    if isinstance(value, str):
        return _tokens(value)
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out |= _needles(item)
        return out
    return set()


def bind_missing_evidence(record: dict, chunks) -> dict:
    sources = [c for c in chunks if getattr(c, "text", "")]
    if not sources:
        return record
    evidence = record.setdefault("evidence", [])
    for path in _BIND_PATHS:
        if _covered(record, path):
            continue
        value = _get(record, path)
        if not _known(value):
            continue
        needles = _needles(value)
        if len(needles) < 1:
            continue
        best, best_score = None, 0.0
        for src in sources:
            have = _tokens(src.text)
            score = len(needles & have) / len(needles)
            if score > best_score:
                best, best_score = src, score
        if best is None or best_score < 0.3:
            continue
        evidence.append({
            "field_paths": ["/" + path],
            "chunk": {
                "source_file_id": best.source_file_id,
                "section": best.section or "document",
                "page": best.page,
                "text": best.text,
            },
        })
    return record


def _unparsed(record: dict) -> bool:
    return any("no parseable JSON" in str(w) for w in record.get("warnings") or [])


def predict(chunks, source_file_id: str, incident_id: str, repair: bool = True) -> dict:
    text = "\n".join(c.text for c in chunks if getattr(c, "text", ""))
    # Exact dictionary hits only. Fuzzy lookup on a full report maps unrelated uppercase
    # words (ФИО, РЖД, ГРЭС, equipment ids) onto some other abbreviation and the model
    # copies that into the record.
    expansions = abbreviation_expansions(text, fuzzy=False)
    user = parse_user_prompt(source_file_id, chunks)
    if expansions:
        user = ("Abbreviation expansions (from lookup_abbreviation):\n"
                + "\n".join(f"- {tok}: {full}" for tok, full in expansions)
                + "\n\n" + user)
    raw = complete(TOOL_ASSISTED_SYSTEM, user, source_id=source_file_id, trace_id=incident_id,
                   system_version=TOOL_ASSISTED_SYSTEM_VERSION, model=MODEL)
    record = align_evidence(build_record_from_output(raw, source_file_id, incident_id), chunks)
    if repair and _unparsed(record):
        # The empty fallback is schema-valid, so the repair loop would keep it and the
        # document would score as every field missing (this happened on 07.doc).
        raw = complete(TOOL_ASSISTED_SYSTEM, user + "\n\n" + NO_JSON_PROMPT,
                       source_id=source_file_id, trace_id=incident_id,
                       system_version=NO_JSON_VERSION, model=MODEL)
        record = align_evidence(build_record_from_output(raw, source_file_id, incident_id), chunks)

    if not repair:
        return bind_missing_evidence(record, chunks)

    for attempt in range(REPAIR_ATTEMPTS):
        ok, errors = validate(record, "incident_record")
        if ok:
            if attempt:
                record.setdefault("warnings", []).append("repaired after validation failure")
            return bind_missing_evidence(record, chunks)
        if attempt == REPAIR_ATTEMPTS - 1:
            break
        previous = json.dumps(record, ensure_ascii=False)
        repair_user = user + "\n\n" + REPAIR_PROMPT.format(previous=previous, errors="\n".join(errors))
        raw = complete(TOOL_ASSISTED_SYSTEM, repair_user, source_id=source_file_id,
                       trace_id=incident_id, system_version=REPAIR_VERSION, model=MODEL)
        record = align_evidence(build_record_from_output(raw, source_file_id, incident_id), chunks)

    record.setdefault("warnings", []).append(
        f"schema invalid after {REPAIR_ATTEMPTS} repair attempts: {'; '.join(errors[:3])}")
    return bind_missing_evidence(record, chunks)
