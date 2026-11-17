_SYSTEMS = None


def answer(system: str, question: str, index, records_by_id: dict, top_k: int = 5, source_id: str = "qa") -> dict:
    global _SYSTEMS
    if _SYSTEMS is None:
        from qa import baseline, direct, tool_assisted
        _SYSTEMS = {
            "baseline": baseline.answer,
            "direct": direct.answer,
            "tool_assisted": tool_assisted.answer,
        }
    return _SYSTEMS[system](question, index, records_by_id, top_k=top_k, source_id=source_id)
