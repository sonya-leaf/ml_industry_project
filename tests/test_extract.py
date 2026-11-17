import docx

from parsing.extract import extract_document


def _make_docx(path: str) -> None:
    d = docx.Document()
    head = d.add_paragraph("Типовая справка о работе устройств РЗА")
    head.runs[0].bold = True
    d.add_paragraph("Вводный текст про подстанцию.")
    t = d.add_table(rows=2, cols=3)
    hdr = t.rows[0].cells
    hdr[0].text = "№"
    hdr[1].text = "Раздел"
    hdr[2].text = "Описание"
    row = t.rows[1].cells
    row[0].text = "1."
    row[1].text = "Присоединение"
    row[2].text = "КВЛ 500 кВ Ногинск"
    d.add_paragraph("Заключительный абзац.")
    d.save(path)


def test_extract_sections_and_table(tmp_path):
    p = str(tmp_path / "rep.docx")
    _make_docx(p)
    chunks = extract_document(p)
    texts = {c.section: c.text for c in chunks}

    # bold heading becomes a section whose text includes the heading
    assert "Типовая справка о работе устройств РЗА" in texts
    assert "Вводный текст про подстанцию." in texts["Типовая справка о работе устройств РЗА"]

    # table "Раздел"/"Описание" row becomes its own section
    assert "Присоединение" in texts
    assert "КВЛ 500 кВ Ногинск" in texts["Присоединение"]

    # trailing paragraph belongs to the last section
    assert any("Заключительный абзац." in v for v in texts.values())

    # page is set (docx with no page breaks -> page 1)
    for c in chunks:
        assert c.page == 1
        assert c.source_file_id == "rep.docx"
        assert c.section
        assert c.text


def test_direct_prompt_does_not_expand_abbreviations():
    from parsing.prompts import DIRECT_SYSTEM, TOOL_ASSISTED_SYSTEM

    assert "Do not expand them" in DIRECT_SYSTEM
    assert "lookup_abbreviation" in TOOL_ASSISTED_SYSTEM


def test_bind_missing_evidence_cites_the_chunk_that_contains_the_value():
    from parsing.extract import DocumentChunk
    from parsing.tool_assisted import bind_missing_evidence

    chunk = DocumentChunk("10.docx", "Оборудование", 1, "АТ-2 500 кВ, сторона 220 кВ и 35 кВ.")
    other = DocumentChunk("10.docx", "Прочее", 1, "Погода на объекте была ясной весь день.")
    record = {"observations": [{"voltage_levels_kv": [500, 220, 35], "facility": None}], "evidence": []}
    bind_missing_evidence(record, [other, chunk])
    paths = [ev["field_paths"] for ev in record["evidence"]]
    assert ["/observations/0/voltage_levels_kv"] in paths
    cited = record["evidence"][0]["chunk"]["text"]
    assert "500" in cited and "220" in cited


def test_align_evidence_replaces_a_paraphrase_and_keeps_a_verbatim_quote():
    from parsing.extract import DocumentChunk
    from parsing.tool_assisted import align_evidence

    src = DocumentChunk("02.docx", "Описание", 1,
                        "Автоматическое повторное включение секции шин было успешным после осмотра.")
    other = DocumentChunk("02.docx", "Прочее", 1, "Совсем другой текст про погоду на объекте вчера.")
    record = {"evidence": [
        {"chunk": {"source_file_id": "x", "section": "?", "page": None,
                   "text": "повторное включение шин секции прошло успешно"}},
        {"chunk": {"source_file_id": "02.docx", "section": "Описание", "page": 1,
                   "text": "было успешным"}},
    ]}
    align_evidence(record, [other, src])
    assert record["evidence"][0]["chunk"]["text"] == src.text
    assert record["evidence"][0]["chunk"]["section"] == "Описание"
    assert record["evidence"][1]["chunk"]["text"] == "было успешным"
