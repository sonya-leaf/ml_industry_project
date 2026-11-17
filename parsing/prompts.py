# --- shared building blocks -------------------------------------------------

SCHEMA_CONTRACT = """\
Output ONLY a JSON object conforming to the IncidentRecord schema (v1.0):
{
  "schema_version": "1.0",
  "incident_id": "<unique id, e.g. 'PARSING-NN'>",
  "source_file_id": "<exact source file id>",
  "report_stage": "primary|clarified|final|unknown",
  "occurred_at": "YYYY-MM-DDTHH:MM:SS" | null,
  "event_type": "short_circuit|abnormal_operation_without_short_circuit|other|unknown",
  "fault": {"description": str|null, "faulted_phases": ["A","B","C"], "ground_involved": bool|null} | null,
  "observations": [{
    "facility": str|null, "voltage_levels_kv": [int], "connections": [str],
    "pre_event_scheme": str|null, "switching_actions": [str], "rpa_operations": [str],
    "reclosing_result": "successful|unsuccessful|mixed|not_attempted|unknown",
    "event_summary": str|null,
    "rpa_assessment": "correct|incorrect|mixed|not_reported",
    "reported_causes": [str], "recommendations": [str]
  }],
  "warnings": [str],
  "evidence": [{
    "field_paths": ["/observations/0/event_summary", ...],
    "chunk": {"source_file_id": str, "section": str, "page": int|null, "text": str}
  }]
}
"""

PARSING_RULES = """\
You are an expert engineer in Russian power-grid relay protection (РЗА) extracting a
structured IncidentRecord from a real incident report (справка or распоряжение).

Rules:
- Use ONLY the provided document chunks. Never invent facts, times, equipment, causes or
  recommendations that are not in the text. When information is absent, use the
  schema-appropriate empty/unknown value (null, [], "unknown", "not_reported").
- "event_type" = "short_circuit" only if a short-circuit (КЗ) is explicitly described;
  otherwise use "abnormal_operation_without_short_circuit" or "unknown".
- Determine "report_stage" from the report wording (первичная/первичный -> "primary",
  завершённая/завершённый or финальный -> "final", уточнённая -> "clarified",
  otherwise "unknown").
- "occurred_at" must be a concrete timestamp present in the report (set null if only a date
  or nothing is given).
- Evidence is mandatory: for EVERY populated field include the exact source chunk
  (source_file_id, section, page, text) and the JSON-pointer field_path. Never fabricate
  evidence text; copy it verbatim from the chunk.
"""


# --- direct (v1: no tools, no abbreviation dictionary) ----------------------
DIRECT_SYSTEM = f"""{PARSING_RULES}
- Copy domain abbreviations (АПВ, ДФЗ, ДЗТ, ТЗНП, ОМП, ...) exactly as they appear in the
  chunks. Do not expand them and do not guess a full form: you have no abbreviation dictionary.

You have NO access to any tools: answer purely from the provided chunks.

{SCHEMA_CONTRACT}"""

DIRECT_SYSTEM_VERSION = "parsing.direct.v2"

# --- tool-assisted (v2) -----------------------------------------------------

TOOL_PROTOCOL = """\
Tool use:
- Abbreviation expansions for this document are supplied in the user message (resolved by the
  lookup_abbreviation tool over data/abbreviations.csv). When you write a fact that uses one
  of those abbreviations, write the full form and then the abbreviation in parentheses
  ("автоматическое повторное включение (АПВ)"). Expand only a fact the report states.
  Do not add a protection, connection, cause or recommendation just because its abbreviation
  is in the list.
- Cite evidence verbatim from the provided chunks (get_document_chunks). Every populated
  field needs a field_path pointing at the chunk that states it.
"""

TOOL_ASSISTED_SYSTEM = f"""{PARSING_RULES}

{TOOL_PROTOCOL}

{SCHEMA_CONTRACT}"""

TOOL_ASSISTED_SYSTEM_VERSION = "parsing.tool_assisted.v2"


def parse_user_prompt(source_file_id: str, chunks) -> str:
    lines = [f"Document: {source_file_id}", "Chunks:"]
    for i, c in enumerate(chunks):
        page = c.page if c.page is not None else "?"
        lines.append(f"[{i}] section={c.section!r} page={page}\n{c.text}")
    return "\n".join(lines)
