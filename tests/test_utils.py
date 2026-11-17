from utils import load_schema, validate


def test_load_schema():
    assert load_schema("incident_record")["title"].startswith("IncidentRecord")


def test_validate_qa_response_resolves_evidence_ref():
    resp = {
        "schema_version": "1.0",
        "status": "answered",
        "answer": "x",
        "citations": [{
            "incident_id": "DEMO-1",
            "evidence": {
                "field_paths": ["/event_type"],
                "chunk": {"source_file_id": "d.docx", "section": "s", "page": 1, "text": "t"},
            },
        }],
    }
    ok, errors = validate(resp, "qa_response")
    assert ok, errors


def test_validate_good_and_bad():
    good = {
        "schema_version": "1.0",
        "incident_id": "X-1",
        "source_file_id": "01.docx",
        "report_stage": "primary",
        "occurred_at": None,
        "event_type": "unknown",
        "fault": None,
        "observations": [],
        "warnings": [],
        "evidence": [],
    }
    ok, errors = validate(good, "incident_record")
    assert ok and errors == []

    bad = dict(good)
    bad["report_stage"] = "bogus"
    ok, errors = validate(bad, "incident_record")
    assert not ok and errors
