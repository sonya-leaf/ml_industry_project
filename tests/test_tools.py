from qa.retrieve import BM25Index
from tools import lookup_abbreviation, query_incidents


def test_lookup_abbreviation_exact_and_fuzzy():
    top = lookup_abbreviation("ДФЗ")
    assert top and top[0]["abbreviation"] == "ДФЗ"
    assert top[0]["full_form"] == "дифференциально-фазная защита"

    # fuzzy: a slightly different spelling still finds the right entry at top
    top_fuzzy = lookup_abbreviation("ДЗТ", top_k=1)
    assert top_fuzzy and top_fuzzy[0]["abbreviation"] == "ДЗТ"


def test_expansions_skip_fuzzy_false_friends_when_disabled():
    from tools import abbreviation_expansions

    text = "Подписал ФИО, объект РЖД, станция ГРЭС"
    assert abbreviation_expansions(text, fuzzy=False) == []
    fuzzy = dict(abbreviation_expansions(text, fuzzy=True))
    assert "ФИО" in fuzzy  # matched to an unrelated dictionary row


def test_lookup_empty_and_unknown():
    assert lookup_abbreviation("") == []
    assert lookup_abbreviation("XYZНЕСУЩЕСТВУЮЩАЯ") == []


def test_query_incidents_requires_index():
    try:
        query_incidents("АПВ")
    except ValueError:
        return
    raise AssertionError("query_incidents should require an index")


def test_query_incidents_merges_multiple_queries():
    idx = BM25Index()
    idx.build([
        ("a", "автоматическое повторное включение АПВ"),
        ("b", "дифференциальная защита ошиновки"),
    ])
    res = query_incidents(["автоматическое повторное включение", "АПВ"], top_k=2, index=idx)
    assert res[0][0] == "a"


def test_query_incidents_pool_keeps_a_hit_below_each_top_k():
    class Idx:
        def search(self, query, top_k=5):
            rows = {
                "q1": [("a", 1.0), ("target", 0.4)],
                "q2": [("b", 1.0), ("target", 0.4)],
            }[query]
            return rows[:top_k]

    narrow = [d for d, _ in query_incidents(["q1", "q2"], top_k=2, index=Idx(), pool=1)]
    wide = [d for d, _ in query_incidents(["q1", "q2"], top_k=3, index=Idx(), pool=2)]
    assert "target" not in narrow
    assert "target" in wide


def test_query_incidents_keeps_one_strong_hit_over_a_repeated_weak_one():
    class Idx:
        def search(self, query, top_k=5):
            rows = {
                "q1": [("strong", 5.0), ("weak", 4.0)],
                "q2": [("weak", 4.0)],
            }[query]
            return rows[:top_k]

    # Summing would rank weak first (4+4 > 5). The best single score ranks strong first.
    assert query_incidents(["q1", "q2"], top_k=1, index=Idx())[0][0] == "strong"
