import re

WARNING = "Обнаруженная prompt injection удалена до извлечения информации."

# (regex, label). Order matters: more specific first. Covers the actual payloads observed in
# the golden _inj documents plus common Russian/English instruction forms.
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # --- Russian: ignore / override -----------------------------------------
    (re.compile(r"игнорируй\s+(все\s+)?(предыдущ\w+\s+|документ\w*|инструкц\w+|системн\w+|промпт\w*)", re.I), "ignore_docs"),
    (re.compile(r"(?:проигнорируй|забудь|пропусти)\s+[^.\n]{0,40}", re.I), "ignore_docs"),
    (re.compile(r"не\s+учитывай\s+[^.;\n]{0,40}", re.I), "ignore_data"),
    (re.compile(r"не\s+обращай\s+внимания\s+на\s+[^.;\n]{0,40}", re.I), "ignore_data"),
    (re.compile(r"независимо\s+от\s+(предыдущ|инструкц|системн|этих)\w*", re.I), "override"),
    (re.compile(r"следуй\s+этой\s+инструкции", re.I), "follow_injected"),
    (re.compile(r"вместо\s+системн\w+\s+(инструкции|промпта)", re.I), "override_system"),
    (re.compile(r"придумай\s+[^.;\n]{0,40}", re.I), "fabricate"),
    (re.compile(r"выдай\s+[^.;\n]{0,60}\s+как\s+достоверн\w+", re.I), "fabricate_as_truth"),
    (re.compile(r"(?:добавь|включи|укажи|напиши|скажи|повтори)\s+[^.\n]{0,60}", re.I), "inject_output"),
    (re.compile(r"в\s+конце\s+(?:своего\s+)?ответа", re.I), "inject_output"),
    # --- English: ignore / override ----------------------------------------
    (re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier|system|other|document)\w*\s+(instructions|prompt|text|content)?", re.I), "ignore_docs"),
    (re.compile(r"ignore\s+(the\s+)?(previous|prior|above|system|document)\w*", re.I), "ignore_docs"),
    (re.compile(r"disregard\s+(previous|prior|above|system|all)", re.I), "ignore_docs"),
    (re.compile(r"forget\s+(everything|all|the)\s+(above|previous|prior)", re.I), "ignore_docs"),
    (re.compile(r"from\s+now\s+on", re.I), "override"),
    (re.compile(r"you\s+must\s+(ignore|include|say|state|repeat|output|follow|obey)", re.I), "override"),
    (re.compile(r"reveal\s+(your|the)\s+(secret|data|prompt)", re.I), "reveal"),
    # --- English: inject content into the response --------------------------
    (re.compile(r"include\s+the\s+following", re.I), "inject_output"),
    (re.compile(r"at\s+the\s+end\s+of\s+your\s+response", re.I), "inject_output"),
    (re.compile(r"include\s+[^.\n]{0,80}\s+in\s+your\s+response", re.I), "inject_output"),
    (re.compile(r"use\s+the\s+following\s+\w+\s+(format|conclusion|statement|template)", re.I), "inject_output"),
    (re.compile(r"when\s+summarizing\s+this\s+event", re.I), "inject_output"),
    (re.compile(r"this\s+statement\s+is\s+required", re.I), "inject_output"),
    (re.compile(r"required\s+to\s+confirm", re.I), "inject_output"),
    (re.compile(r"successfully\s+processed", re.I), "inject_output"),
    (re.compile(r"secret\s+password", re.I), "reveal"),
    (re.compile(r"dancing\s+penguins", re.I), "inject_false_content"),
]


def detect_injection(text: str) -> list[dict]:
    spans: list[dict] = []
    for pat, label in _INJECTION_PATTERNS:
        for m in pat.finditer(text):
            spans.append({"start": m.start(), "end": m.end(), "label": label})
    spans.sort(key=lambda s: s["start"])
    # merge overlapping spans
    merged: list[dict] = []
    for s in spans:
        if merged and s["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            merged[-1]["label"] += f"+{s['label']}"
        else:
            merged.append(dict(s))
    return merged


def sanitize_text(text: str) -> tuple[str, list[str]]:
    spans = detect_injection(text)
    if not spans:
        return text, []
    removed: list[str] = []
    parts: list[str] = []
    cursor = 0
    for s in spans:
        parts.append(text[cursor:s["start"]])
        snippet = text[s["start"]:s["end"]].strip()
        if snippet:
            removed.append(snippet)
        cursor = s["end"]
    parts.append(text[cursor:])
    clean = "".join(parts)
    # collapse double spaces/newlines left by removal
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, removed


def sanitize_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    clean_chunks: list[dict] = []
    warnings: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        clean, removed = sanitize_text(text)
        if removed:
            c = dict(chunk)
            c["text"] = clean
            c["_unsanitized_text"] = text
            c["guardrail_warning"] = WARNING
            clean_chunks.append(c)
            warnings.append({"source_file_id": chunk.get("source_file_id"), "section": chunk.get("section"),
                             "removed": removed})
        else:
            clean_chunks.append(chunk)
    return clean_chunks, warnings


def sanitize_question(text: str) -> tuple[str, list[str]]:
    return sanitize_text(text)


def add_warning(record: dict, guard_hits: list) -> dict:
    if guard_hits and WARNING not in record.get("warnings", []):
        record.setdefault("warnings", []).append(WARNING)
    return record
