import glob
import json
import os
import re
import time

from utils import ROOT, load_config

RUN_ID = os.getenv("PGIA_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
RUN_DATE = time.strftime("%Y%m%d")
_TASKS = {"parsing", "qa", "judge"}
_UNSAFE = re.compile(r"[^\w.\-]+")
_PRICING: dict | None = None


def _pricing() -> dict:
    global _PRICING
    if _PRICING is None:
        _PRICING = load_config().get("pricing") or {}
    return _PRICING


def _cost(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float:
    price = _pricing().get(model)
    if not price:
        return 0.0
    per_in = float(price.get("per_1m_in", 0.0))
    per_out = float(price.get("per_1m_out", 0.0))
    return (prompt_tokens or 0) / 1e6 * per_in + (completion_tokens or 0) / 1e6 * per_out


def _traces_root() -> str:
    return os.path.join(ROOT, load_config()["paths"]["out"], "traces")


def _task(system_version) -> str:
    prefix = str(system_version or "").split(".", 1)[0]
    return prefix if prefix in _TASKS else "other"


def _file_name(item_id) -> str:
    name = _UNSAFE.sub("_", str(item_id or "")).strip("._")
    return f"{RUN_DATE}_{name or 'unknown'}.jsonl"


def log_call(*, system: str, system_version, user: str, output: str, source_id: str | None,
             model: str, latency_s: float, publishable: bool = False,
             trace_id: str | None = None,
             prompt_tokens: int | None = None, completion_tokens: int | None = None) -> None:
    task = _task(system_version)
    entry = {
        "ts": time.time(),
        "run_id": RUN_ID,
        "task": task,
        "model": model,
        "system_version": system_version,
        "source_id": source_id,
        "trace_id": trace_id,
        "publishable": publishable,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(_cost(model, prompt_tokens, completion_tokens), 6),
        "prompt_len_chars": len(system) + len(user),
        "output_len_chars": len(output),
        "system": system,
        "user": user,
        "output": output,
    }
    d = os.path.join(_traces_root(), task)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, _file_name(trace_id or source_id)), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def report(trace_files: list[str] | None = None, run_id: str | None = None) -> dict:
    if trace_files is None:
        trace_files = sorted(glob.glob(os.path.join(_traces_root(), "**", "*.jsonl"), recursive=True))
    total = 0
    total_latency = 0.0
    total_in = 0
    total_out = 0
    total_cost = 0.0
    total_in_tokens = 0
    total_out_tokens = 0
    by_model: dict[str, int] = {}
    by_task: dict[str, int] = {}
    cost_by_model: dict[str, float] = {}
    for path in trace_files:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if run_id and e.get("run_id") != run_id:
                    continue
                total += 1
                total_latency += e.get("latency_s", 0.0)
                total_in += e.get("prompt_len_chars", 0)
                total_out += e.get("output_len_chars", 0)
                total_in_tokens += e.get("prompt_tokens") or 0
                total_out_tokens += e.get("completion_tokens") or 0
                cost = e.get("cost_usd") or 0.0
                total_cost += cost
                m = e.get("model", "?")
                by_model[m] = by_model.get(m, 0) + 1
                cost_by_model[m] = cost_by_model.get(m, 0.0) + cost
                t = e.get("task", "?")
                by_task[t] = by_task.get(t, 0) + 1
    return {
        "calls": total,
        "total_latency_s": round(total_latency, 2),
        "prompt_chars": total_in,
        "output_chars": total_out,
        "prompt_tokens": total_in_tokens,
        "completion_tokens": total_out_tokens,
        "total_cost_usd": round(total_cost, 6),
        "by_model": by_model,
        "by_task": by_task,
        "cost_by_model": {k: round(v, 6) for k, v in cost_by_model.items()},
    }
