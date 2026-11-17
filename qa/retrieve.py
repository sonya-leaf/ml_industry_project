import json
import math
import re
from collections import defaultdict

from tools import abbreviation_expansions
from utils import read_json, write_json

_WORD_RE = re.compile(r"[^\W_]+")

# Russian suffix endings to strip for light stemming (crude but effective for grouping
# grammatical variants: кабель/кабеля/кабелем -> кабел, замена/замене -> замен,
# Мирная/Мирной/Мирной -> мирн). Adjective endings come before the single-letter ones so
# inflected proper-noun forms converge on the same stem.
_STEM_ENDINGS = (
    # multi-letter (adjectival / plural / prepositional)
    "иями", "ями", "ами", "иях", "ях", "ией", "ию", "ия", "ии",
    "ах", "ам", "ом", "ем", "ым", "им", "ых", "их",
    "ую", "ая", "яя", "ое", "ее", "ие", "ые", "ый", "ий", "ой", "ов", "ев", "ей",
    # single-letter
    "а", "я", "ы", "и", "е", "у", "о", "ь",
)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def stem(word: str) -> str:
    for end in _STEM_ENDINGS:
        if word.endswith(end) and len(word) - len(end) >= 3:
            return word[: -len(end)]
    return word


def stem_tokenize(text: str) -> list[str]:
    return [stem(w) for w in tokenize(text)]


# Structural words that appear in every facility name ("подстанция (ПС) 330 кВ Мирная").
# They are not distinctive, so they must not be used to identify a facility in a query.
_FACILITY_STOP = {
    "подстанция", "подстанц", "станция", "станц", "электростанция",
    "пс", "кв", "вл", "эс",
}


def _is_facility_token(tok: str) -> bool:
    return not tok.isdigit() and tok not in _FACILITY_STOP


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer=tokenize):
        self.k1 = k1
        self.b = b
        self._tokenize = tokenizer
        self.postings: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.doc_len: dict[str, int] = {}
        self.doc_count = 0
        self.avgdl = 0.0

    def _reset(self) -> None:
        self.postings.clear()
        self.doc_len.clear()
        self.doc_count = 0
        self.avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        freqs: dict[str, int] = {}
        for tok in self._tokenize(text):
            freqs[tok] = freqs.get(tok, 0) + 1
        for tok, tf in freqs.items():
            self.postings[tok][doc_id] = tf
        self.doc_len[doc_id] = sum(freqs.values())
        self.doc_count = len(self.doc_len)
        self.avgdl = sum(self.doc_len.values()) / self.doc_count if self.doc_count else 0.0

    def build(self, documents) -> None:
        self._reset()
        for doc_id, text in documents:
            self.add(doc_id, text)

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avgdl": self.avgdl,
            "doc_len": self.doc_len,
            "postings": {term: dict(docs) for term, docs in self.postings.items()},
        }
        write_json(path, payload)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        payload = read_json(path)
        idx = cls(k1=payload["k1"], b=payload["b"])
        idx.doc_count = payload["doc_count"]
        idx.avgdl = payload["avgdl"]
        idx.doc_len = {k: int(v) for k, v in payload["doc_len"].items()}
        for term, docs in payload["postings"].items():
            idx.postings[term] = defaultdict(int, {k: int(v) for k, v in docs.items()})
        return idx

    # ---- retrieval ---------------------------------------------------------

    def _idf(self, term: str) -> float:
        df = len(self.postings[term])
        if df == 0:
            return 0.0
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        terms = self._tokenize(query)
        if not terms or not self.doc_count:
            return []

        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            idf = self._idf(term)
            if idf == 0:
                continue
            for doc_id, tf in self.postings[term].items():
                dl = self.doc_len.get(doc_id, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


def _unique_chunks(rec: dict) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for e in rec.get("evidence", []):
        chunk = e.get("chunk") or {}
        text = chunk.get("text", "")
        if text and text not in seen:
            seen.add(text)
            out.append(chunk)
    return out


def build_incident_corpus(records: list[dict], tokenizer=tokenize) -> tuple["BM25Index", dict[str, list[dict]]]:
    index = BM25Index(tokenizer=tokenizer)
    chunk_store: dict[str, list[dict]] = {}
    for rec in records:
        iid = rec.get("incident_id", "")
        if not iid:
            continue
        unique = _unique_chunks(rec)
        index.add(iid, " ".join(c.get("text", "") for c in unique))
        chunk_store[iid] = unique
    return index, chunk_store


def records_context(records_by_id: dict[str, dict], ids: list[str]) -> str:
    parts = []
    for iid in ids:
        rec = records_by_id.get(iid)
        if not rec:
            continue
        parts.append(f"[{iid}]\n" + json.dumps(rec, ensure_ascii=False))
    return "\n\n".join(parts)


def index_records(records: list[dict]) -> tuple["HybridRetriever", dict[str, dict]]:
    return HybridRetriever(records), {r["incident_id"]: r for r in records}

# --- retrieval improvements (chunk-level + abbreviation expansion) ------------------------


def expand_query(query: str) -> str:
    if not query:
        return query
    additions = [full for _, full in abbreviation_expansions(query)]
    return query + (" " + " ".join(additions) if additions else "")


def build_chunk_index(records: list[dict], tokenizer=tokenize) -> tuple["BM25Index", dict[str, str]]:
    index = BM25Index(tokenizer=tokenizer)
    parent: dict[str, str] = {}
    for rec in records:
        iid = rec.get("incident_id", "")
        if not iid:
            continue
        for k, chunk in enumerate(_unique_chunks(rec)):
            cid = f"{iid}::{k}"
            index.add(cid, chunk.get("text", ""))
            parent[cid] = iid
    return index, parent


def extract_incident_ids(query: str, known_ids) -> list[str]:
    nums = [int(n) for n in re.findall(r"QA[-_ ]?(\d+)", query or "", re.IGNORECASE)]
    if not nums:
        return []
    hits: list[str] = []
    seen: set[int] = set()
    for n in nums:
        if n in seen:
            continue
        seen.add(n)
        for kid in known_ids:
            if any(int(t) == n for t in re.findall(r"\d+", kid)):
                hits.append(kid)
                break
    return hits


def extract_facility_ids(query: str, facility_terms: dict[str, list[str]], tokenizer) -> list[str]:
    if not facility_terms:
        return []
    hits: list[str] = []
    seen: set[str] = set()
    for tok in tokenizer(query):
        if not _is_facility_token(tok):
            continue
        for iid in facility_terms.get(tok, ()):
            if iid not in seen:
                seen.add(iid)
                hits.append(iid)
    return hits


class HybridRetriever:
    def __init__(self, records: list[dict], expand: bool = True, stem: bool = True):
        self.expand = expand
        self._ids = [r.get("incident_id") for r in records if r.get("incident_id")]
        tok = stem_tokenize if stem else tokenize
        self._tok = tok
        self._facility_terms: dict[str, list[str]] = {}
        for rec in records:
            iid = rec.get("incident_id")
            if not iid:
                continue
            for obs in rec.get("observations", []):
                fac = obs.get("facility")
                if fac:
                    for t in tok(fac):
                        if not _is_facility_token(t):
                            continue
                        lst = self._facility_terms.setdefault(t, [])
                        if iid not in lst:
                            lst.append(iid)
        self.doc_index, self.chunk_store = build_incident_corpus(records, tokenizer=tok)
        self.chunk_index, self.parent = build_chunk_index(records, tokenizer=tok)

    def _query(self, query: str, expand: bool | None = None) -> str:
        return expand_query(query) if (self.expand if expand is None else expand) else query

    def search(self, query: str, top_k: int = 5, doc_weight: float = 1.0, chunk_weight: float = 1.0,
               expand: bool | None = None, lookup: bool = True) -> list[tuple[str, float]]:

        q = self._query(query, expand)
        broad = max(top_k, 20)
        scores: dict[str, float] = defaultdict(float)
        for iid, s in self.doc_index.search(q, top_k=broad):
            scores[iid] += doc_weight * s
        if chunk_weight:
            for cid, s in self.chunk_index.search(q, top_k=broad):
                iid = self.parent.get(cid)
                if iid:
                    scores[iid] += chunk_weight * s
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        if not lookup:
            return ranked[:top_k]

        forced: list[str] = []
        seen_forced: set[str] = set()
        for iid in extract_incident_ids(query, self._ids):
            if iid not in seen_forced:
                seen_forced.add(iid)
                forced.append(iid)
        for iid in extract_facility_ids(query, self._facility_terms, self._tok):
            if iid not in seen_forced:
                seen_forced.add(iid)
                forced.append(iid)
        if not forced:
            return ranked[:top_k]

        # forced (explicit/facility) ids always outrank BM25; a large finite score keeps them
        # first when query_incidents merges by the best score.
        forced_score = 1e9
        seen = set(forced)
        out = [(iid, forced_score) for iid in forced]
        for iid, s in ranked:
            if iid in seen:
                continue
            out.append((iid, s))
            seen.add(iid)
            if len(out) >= top_k:
                break
        return out[:top_k]
