_SYSTEMS = None


def parse(system: str, chunks, source_file_id: str, incident_id: str) -> dict:
    global _SYSTEMS
    if _SYSTEMS is None:
        from parsing import baseline, direct, tool_assisted
        _SYSTEMS = {
            "baseline": baseline.parse_baseline,
            "direct": direct.predict,
            "tool_assisted": tool_assisted.predict,
        }
    return _SYSTEMS[system](chunks, source_file_id, incident_id)
