import glob

from evaluation import (
    f1_at_k,
    mean_metric,
    parse_ids_list,
    precision_at_k,
    recall_at_k,
    schema_valid_rate,
)
from utils import read_json


def test_recall_precision_f1():
    pred = ["a", "b", "c", "d"]
    gold = ["a", "c"]
    assert recall_at_k(pred, gold, k=2) == 0.5
    assert precision_at_k(pred, gold, k=2) == 0.5
    assert f1_at_k(pred, gold, k=2) == 0.5
    assert recall_at_k(pred, gold) == 1.0  # k=None -> over all predictions


def test_empty_gold():
    assert recall_at_k(["a"], []) == 0.0
    assert precision_at_k(["a"], [], k=1) == 0.0


def test_mean_metric():
    pairs = [(["a"], ["a"]), (["x"], ["a"])]
    assert mean_metric(pairs, recall_at_k, k=1) == 0.5


def test_parse_ids_list():
    assert parse_ids_list('["QA-02","QA-03"]') == ["QA-02", "QA-03"]
    assert parse_ids_list("") == []
    assert parse_ids_list("not json") == []


def test_schema_valid_rate_on_gold():
    objs = [read_json(p) for p in sorted(glob.glob("data/golden_set/parsed_docs/parsing/*.json"))]
    rate, n = schema_valid_rate(objs, "incident_record")
    assert n == len(objs)  # the committed gold records must be schema-valid
    assert rate == 1.0


def test_field_f1_from_judge_credits_semantic_correctness():
    from evaluation import field_f1_from_judge

    gold = {"occurred_at": "2024-08-29T11:22:00", "event_type": "short_circuit", "observations": []}
    pred = {"occurred_at": "2024-08-29 11:22", "event_type": "short_circuit", "observations": []}
    # judge says both are semantically correct even though occurred_at differs in format
    verdicts = {"occurred_at": "correct", "event_type": "correct"}

    per_field, macro = field_f1_from_judge([verdicts], [gold], [pred])
    assert macro == 1.0
    assert per_field["occurred_at"]["f1"] == 1.0

    # a judge "incorrect" on a known field becomes both FP and FN
    verdicts_bad = {"occurred_at": "incorrect", "event_type": "correct"}
    _, macro_bad = field_f1_from_judge([verdicts_bad], [gold], [pred])
    assert per_field["occurred_at"]["f1"] == 1.0
    assert macro_bad < 1.0


def test_citation_grounding():
    from evaluation import citation_grounding

    records_by_id = {"QA-01": {"incident_id": "QA-01", "evidence": [
        {"chunk": {"text": "В 11:55 произошло короткое замыкание."}},
        {"chunk": {"text": "АПВ сработало."}},
    ]}}

    grounded = {"citations": [{"incident_id": "QA-01", "evidence": {"chunk": {"text": "короткое замыкание."}}}]}
    fabricated = {"citations": [{"incident_id": "QA-01", "evidence": {"chunk": {"text": "такого текста нет"}}}]}
    mixed = {"citations": [
        {"incident_id": "QA-01", "evidence": {"chunk": {"text": "АПВ сработало."}}},
        {"incident_id": "QA-01", "evidence": {"chunk": {"text": "выдуманный факт"}}},
    ]}

    assert citation_grounding([grounded], records_by_id) == 1.0
    assert citation_grounding([fabricated], records_by_id) == 0.0
    assert citation_grounding([mixed], records_by_id) == 0.5
    assert citation_grounding([{"citations": []}], records_by_id) == 1.0
