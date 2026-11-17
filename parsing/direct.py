from llm import DEFAULT_MODEL, complete
from parsing.prompts import DIRECT_SYSTEM, DIRECT_SYSTEM_VERSION, parse_user_prompt
from utils import extract_json_object

MODEL = DEFAULT_MODEL


def build_record_from_output(raw: str, source_file_id: str, incident_id: str) -> dict:
    obj = extract_json_object(raw)
    if obj is None:
        return {
            "schema_version": "1.0",
            "incident_id": incident_id,
            "source_file_id": source_file_id,
            "report_stage": "unknown",
            "occurred_at": None,
            "event_type": "unknown",
            "fault": None,
            "observations": [],
            "warnings": ["model output contained no parseable JSON"],
            "evidence": [],
        }
    obj["schema_version"] = "1.0"
    obj["incident_id"] = incident_id
    obj["source_file_id"] = source_file_id
    return obj


def predict(chunks, source_file_id: str, incident_id: str) -> dict:
    user = parse_user_prompt(source_file_id, chunks)
    raw = complete(DIRECT_SYSTEM, user, source_id=source_file_id, trace_id=incident_id,
                   system_version=DIRECT_SYSTEM_VERSION, model=MODEL)
    return build_record_from_output(raw, source_file_id, incident_id)
