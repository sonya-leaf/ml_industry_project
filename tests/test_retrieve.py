from qa.retrieve import (
    BM25Index,
    HybridRetriever,
    build_chunk_index,
    expand_query,
    tokenize,
)


def test_tokenize_cyrillic_and_digits():
    assert tokenize("КВЛ 500 кВ Ногинск-Бескудниково") == ["квл", "500", "кв", "ногинск", "бескудниково"]
    assert tokenize("") == []


def test_search_ranks_relevant_first():
    idx = BM25Index()
    idx.build([
        ("doc1", "дифференциальная защита линии отключила повреждение"),
        ("doc2", "автоматическое повторное включение АПВ вернуло линию"),
        ("doc3", "защита линии сработала правильно"),
    ])
    top = idx.search("дифференциальная защита линии", top_k=2)
    assert top[0][0] == "doc1"
    # doc3 mentions "защита линии" too and should outrank doc2 for this query
    assert {d for d, _ in top} == {"doc1", "doc3"}


def test_empty_collection():
    idx = BM25Index()
    assert idx.search("anything", top_k=5) == []


def test_save_load_roundtrip(tmp_path):
    idx = BM25Index()
    idx.build([("a", "линия отключилась"), ("b", "автотрансформатор включился")])
    p = str(tmp_path / "idx.json")
    idx.save(p)
    loaded = BM25Index.load(p)
    assert loaded.search("линия")[0][0] == "a"
    assert loaded.doc_count == 2


# ---- chunk index + hybrid retrieval ----------------------------------------------

def _record(iid, chunks):
    return {
        "incident_id": iid,
        "evidence": [{"field_paths": ["/observations/0"], "chunk": {"text": t}} for t in chunks],
    }


def test_build_chunk_index_parents_chunks_to_incident():
    recs = [_record("QA-01", ["короткое замыкание фазы B на землю", "отключено защитой линии"]),
            _record("QA-02", ["газовая защита сработала ложно"])]
    cidx, parent = build_chunk_index(recs)
    assert parent["QA-01::0"] == "QA-01"
    assert parent["QA-01::1"] == "QA-01"
    assert parent["QA-02::0"] == "QA-02"
    hits = [d for d, _ in cidx.search("короткое замыкание фазы B", top_k=3)]
    assert hits[0].startswith("QA-01")


def test_hybrid_retriever_finds_incident_from_a_single_section():
    # Whole-record text is noisy; the relevant fact lives in one short chunk.
    recs = [
        _record("QA-01", ["прочие данные по станции", "АПВ 1-й секции неуспешное, АПВ 2-й секции успешное", "ещё данные"]),
        _record("QA-02", ["повреждение ошиновки фазы A"]),
    ]
    hr = HybridRetriever(recs, expand=False)
    top = [d for d, _ in hr.search("АПВ 1-й секции неуспешное", top_k=2)]
    assert "QA-01" in top


def test_expand_query_adds_abbreviation_full_forms():
    expanded = expand_query("Что с АПВ и ДФЗ?")
    assert "автоматическое повторное включение" in expanded
    assert "дифференциально-фазная" in expanded or "дифференциальная" in expanded


def test_expand_query_unknown_token_kept():
    assert "XYZXYZ" in expand_query("про XYZXYZ тут")


def test_stem_tokenize_groups_morphological_variants():
    from qa.retrieve import stem_tokenize
    assert stem_tokenize("кабель") == stem_tokenize("кабеля")
    assert stem_tokenize("замена") == stem_tokenize("замене")
    assert "кабел" in stem_tokenize("кабелем")


def test_stemmed_index_matches_question_morphology():
    from qa.retrieve import BM25Index, stem_tokenize
    idx = BM25Index(tokenizer=stem_tokenize)
    idx.build([("QA-20", "заменить и опрессовать наконечник провода в схеме")])
    assert idx.search("замена наконечниками проводов", top_k=1)[0][0] == "QA-20"


def test_extract_incident_ids_matches_numeric_reference():
    from qa.retrieve import extract_incident_ids
    known = ["QA-02", "QA-13", "PARSING-03_inj", "PARSING-04"]
    assert extract_incident_ids("Что за КЗ было по QA-03?", known) == ["PARSING-03_inj"]
    assert extract_incident_ids("по QA-04 и QA-02 и QA-13", known) == ["PARSING-04", "QA-02", "QA-13"]
    assert extract_incident_ids("ни о чём тут", known) == []


def test_lookup_disabled_does_not_force_an_explicit_id():
    recs = [
        _record("QA-03", ["короткое замыкание фазы B на землю на линии Победа"]),
        _record("QA-06", ["прочие данные по станции и сетке"]),
        _record("QA-12", ["разные данные по ТЭЦ"]),
        _record("QA-17", ["ещё какие-то данные"]),
        _record("QA-16", ["данные про линии"]),
        _record("QA-18", ["некоторые данные"]),
    ]
    hr = HybridRetriever(recs, expand=True, stem=True)
    query = "Что за КЗ было по QA-03 и почему запретили АПВ?"
    assert [d for d, _ in hr.search(query, top_k=5)][0] == "QA-03"
    assert "QA-03" not in [d for d, _ in hr.search(query, top_k=5, expand=False, lookup=False)]


def test_explicit_reference_always_retrieved_even_when_bm25_misses():
    recs = [
        _record("QA-03", ["короткое замыкание фазы B на землю на линии Победа"]),
        _record("QA-06", ["прочие данные по станции и сетке"]),
        _record("QA-12", ["разные данные по ТЭЦ"]),
        _record("QA-17", ["ещё какие-то данные"]),
        _record("QA-16", ["данные про линии"]),
        _record("QA-18", ["некоторые данные"]),
    ]
    hr = HybridRetriever(recs, expand=True, stem=True)
    # query names QA-03 but its terms overlap other docs; explicit ref must surface QA-03
    top = [d for d, _ in hr.search("Что за КЗ было по QA-03 и почему запретили АПВ?", top_k=5)]
    assert top[0] == "QA-03"


def _record_with_facility(iid, facility, text):
    return {
        "incident_id": iid,
        "observations": [{"facility": facility}],
        "evidence": [{"field_paths": ["/observations/0/facility"], "chunk": {"text": text}}],
    }


def test_facility_mention_surfaces_incident():
    recs = [
        _record_with_facility("QA-04", "ПС 220 кВ Мирная", "сетка упала на ошиновку 220 кВ фазы A"),
        _record_with_facility("QA-12", "ТЭЦ-26", "ДФЗ-504 излишне отключил линию"),
        _record_with_facility("QA-17", "ПС Вичуга", "обрыв шлейфа фазы B"),
        _record_with_facility("QA-18", "ПС 500 кВ Старый Оскол", "4 ступень ДЗ отключила линию"),
        _record_with_facility("QA-11", "ПС 500 кВ Старый Оскол", "перекрытие шлейфа"),
    ]
    hr = HybridRetriever(recs, expand=True, stem=True)
    top = [d for d, _ in hr.search("Почему на Мирной АПВ включилось на повреждение?", top_k=5)]
    assert top[0] == "QA-04"
