import re

_DATETIME_RE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})[^\d]{0,4}?в\s*(\d{1,2}):(\d{2})(?::(\d{2}))?")
_DATETIME_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?")
_FACILITY_RE = re.compile(r"ПС\s+(\d+\s*кВ\s+)?([А-ЯЁа-яё0-9][А-ЯЁа-яё0-9\-\s]{2,40}?)(?=[.,;:\n]|$)", re.I)
_KV_RE = re.compile(r"(\d{2,3})\s*кВ")
_PHASE_RE = re.compile(r"фаз[аыи]?\s*[«\"']?\s*([ABC])|ф\.\s*[«\"']?\s*([ABC])|[«\"']([ABC])[«\"']\s*$", re.I)
_GROUND_RE = re.compile(r"на\s+землю|замыкан\w+\s+на\s+землю|земл\w+\s+замыкан", re.I)


def _chunks_text(chunks) -> str:
    return "\n".join(c.text for c in chunks)


def _occurred_at(text: str) -> str | None:
    m = _DATETIME_ISO_RE.search(text) or _DATETIME_RE.search(text)
    if not m:
        return None
    if m.re is _DATETIME_ISO_RE:
        return f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6] or '00'}"
    d, mo, y = int(m[1]), int(m[2]), int(m[3])
    return f"{y:04d}-{mo:02d}-{d:02d}T{int(m[4]):02d}:{int(m[5]):02d}:{m[6] or '00'}"


def _report_stage(text: str) -> str:
    if re.search(r"первичн", text, re.I):
        return "primary"
    if re.search(r"заверш|финальн|уточнённ|уточненн", text, re.I):
        return "final" if re.search(r"заверш|финальн", text, re.I) else "clarified"
    return "unknown"


def _event_type(text: str) -> str:
    if re.search(r"короткое\s+замыкан|КЗ\b|к\.з\.", text, re.I):
        return "short_circuit"
    return "unknown"


def _fault(text: str) -> dict | None:
    m = re.search(r"(?:произошло\s+)?(?:короткое\s+замыкан\w+|перекрыти\w+|обрыв\w+|повреждени\w+|пробо\w+)[^.\n]{0,120}", text, re.I)
    if not m:
        return None
    description = m.group(0).strip()
    phases = [g for g in re.findall(r"фаз[аыи]?\s*[«\"']?\s*([ABC])|ф\.\s*[«\"']?\s*([ABC])", text, re.I) for g in g if g]
    if not phases:
        phases = list({c for c in re.findall(r"\b([ABC])\b", text) if c in "ABC"})
    ground = bool(_GROUND_RE.search(text))
    return {"description": description, "faulted_phases": phases, "ground_involved": ground}


def _summary(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 60]
    if not lines:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if re.search(r"отключ|сработа|АПВ|защит", line, re.I):
            return line
    return lines[0] if lines else None


def _voltage_levels(text: str) -> list[int]:
    return sorted({int(v) for v in _KV_RE.findall(text)}, reverse=True)


def _facility(text: str) -> str | None:
    m = _FACILITY_RE.search(text)
    if not m:
        return None
    name = m.group(2).strip()
    # trim trailing fragments
    name = re.sub(r"\s+на\s+.+$", "", name)
    return ("подстанция (ПС) " + name) if name else None


def parse_baseline(chunks, source_file_id: str, incident_id: str) -> dict:
    text = _chunks_text(chunks)

    occurred = _occurred_at(text)
    facility = _facility(text)
    summary = _summary(text)
    fault = _fault(text)
    stage = _report_stage(text)
    etype = _event_type(text)
    kv = _voltage_levels(text)

    obs = {
        "facility": facility,
        "voltage_levels_kv": kv,
        "connections": [],
        "pre_event_scheme": None,
        "switching_actions": [],
        "rpa_operations": [],
        "reclosing_result": "unknown",
        "event_summary": summary,
        "rpa_assessment": "not_reported",
        "reported_causes": [],
        "recommendations": [],
    }

    # evidence for the fields we managed to populate
    evidence = []
    for path, value in [("/occurred_at", occurred), ("/observations/0/facility", facility),
                        ("/observations/0/event_summary", summary)]:
        if not value:
            continue
        chunk = _find_chunk(chunks, value, occurred)
        if chunk:
            evidence.append({"field_paths": [path], "chunk": chunk.to_dict()})

    return {
        "schema_version": "1.0",
        "incident_id": incident_id,
        "source_file_id": source_file_id,
        "report_stage": stage,
        "occurred_at": occurred,
        "event_type": etype,
        "fault": fault,
        "observations": [obs],
        "warnings": [],
        "evidence": evidence,
    }


def _find_chunk(chunks, *needles):
    for c in chunks:
        low = c.text.lower()
        if any(n and str(n).lower() in low for n in needles):
            return c
    return None
