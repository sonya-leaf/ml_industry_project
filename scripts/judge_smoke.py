import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import judge_parsing_record, judge_qa_answer

gold = {
    "source_file_id": "DEMO-007-final.docx",
    "fault": {"description": "короткое замыкание на землю", "faulted_phases": ["A"], "ground_involved": True},
    "observations": [{"facility": "подстанция (ПС) Северная", "reclosing_result": "unsuccessful",
                      "reported_causes": ["перекрытие изоляции"]}],
    "evidence": [],
}
pred = {
    "source_file_id": "DEMO-007-final.docx",
    "fault": {"description": "КЗ на землю фазы A", "faulted_phases": ["A"], "ground_involved": True},
    "observations": [{"facility": "ПС Северная", "reclosing_result": "successful",
                      "reported_causes": ["перекрытие изоляции"]}],
    "evidence": [
        {"field_paths": ["/fault"], "chunk": {"source_file_id": "DEMO-007-final.docx", "section": "Описание события",
         "page": 1, "text": "В 11:55 произошло короткое замыкание на землю на фазе A ВЛ 220 кВ."}},
        {"field_paths": ["/observations/0/facility"], "chunk": {"source_file_id": "DEMO-007-final.docx",
         "section": "Описание события", "page": 1, "text": "На ПС Северная отключился выключатель Q4."}},
    ],
}

v = judge_parsing_record(gold, pred)
expected_value = {"fault.description": "correct", "observations/0/facility": "correct",
                  "observations/0/reclosing_result": "incorrect", "observations/0/reported_causes": "correct"}
for path, want in expected_value.items():
    print(f"{path:36s} value={v['field_verdicts_value'].get(path):10s} "
          f"with_evidence={v['field_verdicts'].get(path):10s} expected_value={want}")
print("judge_missing", v["judge_missing"])

row = {"question_id": "DEMO-Q1", "expected_status": "answered",
       "reference_answer": "ДЗЛ сработала излишне при внешнем КЗ на землю фазы A из-за некорректного учёта токовых цепей обходного выключателя."}
resp = {"schema_version": "1.0", "status": "answered",
        "answer": "Дифференциальная защита линии сработала излишне при внешнем КЗ; причина — некорректный учёт токовых цепей обходного выключателя.",
        "citations": [{"incident_id": "DEMO-011", "evidence": {"field_paths": ["/observations/0/reported_causes"],
                       "chunk": {"source_file_id": "DEMO-011-final.docx", "section": "Причина", "page": 4,
                                 "text": "При внешнем КЗ зафиксировано излишнее срабатывание ДЗЛ. Причиной стал некорректный учёт токовых цепей обходного выключателя."}}}]}
print(judge_qa_answer("Почему ДЗЛ сработала неправильно?", row, resp))
