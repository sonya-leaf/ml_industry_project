from __future__ import annotations

import json
import os
import time

import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
JUDGE_MODEL = "mistral:7b-instruct-v0.3-q5_K_M"

last_usage: dict = {}


def _endpoint() -> str:
    return os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/chat")


class OllamaJudgeBackend:
    def __init__(self) -> None:
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "300"))
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))

    def complete(self, system: str, user: str, **kw) -> str:
        payload = {
            "model": kw.get("model") or JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": float(kw.get("temperature", 0.0)),
                "num_predict": int(kw.get("max_tokens", 1024)),
                "num_ctx": int(kw.get("num_ctx", 16384)),
                "seed": int(kw.get("seed", 0)),
            },
        }
        if kw.get("json_object", True):
            payload["format"] = "json"

        url = _endpoint()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = (result.get("message") or {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Ollama returned an empty judge response")
                last_usage.clear()
                last_usage.update({
                    "model": payload["model"],
                    "prompt_tokens": result.get("prompt_eval_count"),
                    "completion_tokens": result.get("eval_count"),
                    "finish_reason": "length" if result.get("done_reason") == "length" else result.get("done_reason"),
                })
                return content.strip()
            except urllib.error.HTTPError as exc:
                err = exc.read().decode("utf-8", errors="replace")
                if exc.code == 404:
                    raise RuntimeError(
                        f"Ollama has no model {payload['model']}. "
                        f"Run: ollama pull {payload['model']}\n{err[:500]}"
                    ) from exc
                last_error = RuntimeError(f"Ollama HTTP {exc.code}: {err[:1500]}")
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise RuntimeError(f"Ollama judge failed at {url}: {last_error}") from last_error
        raise RuntimeError("Ollama judge failed")


_backend = OllamaJudgeBackend()


def complete(system: str, user: str, *, source_id: str | None = None, **kw) -> str:
    started = time.time()
    text = _backend.complete(system, user, **kw)
    from logs import trace
    trace.log_call(
        system=system,
        system_version=kw.get("system_version"),
        user=user,
        output=text,
        source_id=source_id,
        model=str(kw.get("model") or JUDGE_MODEL),
        latency_s=round(time.time() - started, 3),
        trace_id=kw.get("trace_id"),
        prompt_tokens=last_usage.get("prompt_tokens"),
        completion_tokens=last_usage.get("completion_tokens"),
    )
    return text
