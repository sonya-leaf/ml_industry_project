import re

from rapidfuzz import fuzz

from utils import load_config, read_csv_rows, repo_path

_ABBR_CACHE: list[dict] | None = None
_ABBR_MAP: dict[str, str] | None = None

# An uppercase run: an abbreviation candidate (АПВ, ДФЗ, КЗ, …).
_UPPER_TOKEN = re.compile(r"[А-ЯЁA-Z][А-ЯЁA-Z0-9]{1,6}")


def load_abbreviations() -> list[dict]:
    global _ABBR_CACHE
    if _ABBR_CACHE is None:
        cfg = load_config()
        path = repo_path(cfg["paths"]["data"], "abbreviations.csv")
        _ABBR_CACHE = read_csv_rows(path)
    return _ABBR_CACHE


def abbreviation_map() -> dict[str, str]:
    global _ABBR_MAP
    if _ABBR_MAP is None:
        mapping: dict[str, str] = {}
        for row in load_abbreviations():
            full = row.get("full_form", "") or ""
            for cand in (row.get("abbreviation", ""), *(row.get("spelling_variants") or "").split(",")):
                cand = cand.strip().upper()
                if cand:
                    mapping[cand] = full
        _ABBR_MAP = mapping
    return _ABBR_MAP


def abbreviation_expansions(text: str, fuzzy: bool = True) -> list[tuple[str, str]]:
    if not text:
        return []
    mapping = abbreviation_map()
    out: list[tuple[str, str]] = []
    for tok in dict.fromkeys(_UPPER_TOKEN.findall(text)):
        upper = tok.upper()
        if upper in mapping:
            out.append((tok, mapping[upper]))
            continue
        if not fuzzy:
            continue
        cands = lookup_abbreviation(upper, context=text[:300], top_k=1)
        if cands:
            full = cands[0].get("full_form")
            if full:
                out.append((tok, full))
    return out


def lookup_abbreviation(query: str, context: str = "", top_k: int = 3) -> list[dict]:
    rows = load_abbreviations()
    if not query:
        return []

    scored: list[tuple[float, dict]] = []
    q = query.strip()
    for row in rows:
        cands = [row.get("abbreviation", ""), *(row.get("spelling_variants") or "").split(",")]
        best = max((fuzz.ratio(q.lower(), c.strip().lower()) / 100.0 for c in cands if c.strip()), default=0.0)
        if best >= 0.4:
            scored.append((best, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [dict(r) for _, r in scored[:top_k]]
    for o, (s, _) in zip(out, scored[:top_k]):
        o["score"] = round(s, 3)
    return out


def query_incidents(queries, top_k: int = 5, index=None, pool: int | None = None) -> list[tuple[str, float]]:
    if index is None:
        raise ValueError("query_incidents requires a BM25 index (pass index=...)")
    if isinstance(queries, str):
        queries = [queries]
    queries = [q for q in queries if q and str(q).strip()]
    depth = max(top_k, pool or top_k)

    merged: dict[str, float] = {}
    for q in queries:
        for doc_id, score in index.search(q, top_k=depth):
            prev = merged.get(doc_id)
            if prev is None or score > prev:
                merged[doc_id] = score
    return sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def get_document_chunks(document_id: str, chunk_store: dict[str, list[dict]] | None = None) -> list[dict]:
    if chunk_store is None:
        return []
    return chunk_store.get(document_id, [])
