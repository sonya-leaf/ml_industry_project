from llm import DEFAULT_MODEL, complete
from qa.baseline import build_response
from qa.prompts import (
    DIRECT_SYSTEM,
    DIRECT_SYSTEM_VERSION,
    REWRITE_SYSTEM,
    REWRITE_VERSION,
    qa_user_prompt,
)
from qa.retrieve import records_context

MODEL = DEFAULT_MODEL


def answer(question: str, index, records_by_id: dict, top_k: int = 5, source_id: str = "qa") -> dict:
    rewritten = complete(REWRITE_SYSTEM, f"Question: {question}", source_id=source_id,
                         system_version=REWRITE_VERSION, model=MODEL).strip()
    # Document BM25 only. expand/chunk/lookup are the tool-assisted retriever; a plain
    # BM25Index (no such arguments) is searched as-is.
    query = rewritten or question
    try:
        hits = index.search(query, top_k=top_k, expand=False, chunk_weight=0.0, lookup=False)
    except TypeError:
        hits = index.search(query, top_k=top_k)
    ids = [doc_id for doc_id, _ in hits]
    ctx = records_context(records_by_id, ids)
    raw = complete(DIRECT_SYSTEM, qa_user_prompt(question, ctx), source_id=source_id,
                   system_version=DIRECT_SYSTEM_VERSION, model=MODEL)
    return build_response(raw)
