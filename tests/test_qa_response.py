from qa.baseline import build_response

EVIDENCE = {"field_paths": ["/observations/0/event_summary"],
            "chunk": {"source_file_id": "a.docx", "section": "s", "page": 1, "text": "текст"}}
ANSWERED = '{"schema_version":"1.0","status":"answered","answer":"да","citations":[{"incident_id":"QA-01","evidence":%s}]}'


def test_valid_passthrough():
    r = build_response(ANSWERED % str(EVIDENCE).replace("'", '"'))
    assert r["status"] == "answered"
    assert len(r["citations"]) == 1


def test_no_json_downgrades():
    r = build_response("просто текст без json")
    assert r["status"] == "insufficient_evidence"
    assert r["citations"] == []


def test_invalid_status_with_answer_is_answered():
    # an unrecognized status with a real answer must not be thrown away
    r = build_response('{"status":"maybe","answer":"x","citations":[]}')
    assert r["status"] == "answered"
    assert r["answer"] == "x"


def test_answered_without_citations_is_kept():
    r = build_response('{"schema_version":"1.0","status":"answered","answer":"да","citations":[]}')
    assert r["status"] == "answered"
    assert r["answer"] == "да"


def test_imperfect_citation_evidence_keeps_answer():
    # missing field_paths in evidence must not downgrade a good answer to insufficient_evidence
    r = build_response('{"schema_version":"1.0","status":"answered","answer":"да",'
                       '"citations":[{"incident_id":"QA-01","evidence":{"chunk":'
                       '{"source_file_id":"a","section":"s","page":1,"text":"t"}}}]}')
    assert r["status"] == "answered"
    assert r["answer"] == "да"


def test_citations_keep_incident_id():
    r = build_response('{"schema_version":"1.0","status":"answered","answer":"да",'
                       '"citations":[{"incident_id":"QA-01"},{"incident_id":"QA-02","evidence":%s}]}'
                       % str(EVIDENCE).replace("'", '"'))
    assert r["status"] == "answered"
    assert [c["incident_id"] for c in r["citations"]] == ["QA-01", "QA-02"]


def test_extra_field_is_stripped():
    # additionalProperties:false must drop unknown keys
    r = build_response('{"schema_version":"1.0","status":"insufficient_evidence","answer":"","citations":[],"extra":1}')
    assert "extra" not in r
