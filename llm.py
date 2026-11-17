import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# The agent model used across parsing and QA variants.
DEFAULT_MODEL = "qwen3.8-27b"


class OpenAIBackend:
    def __init__(self, profile: str = "qwen", **kw):
        self.profile = profile
        self.timeout = kw.get("timeout", 60)
        self.max_retries = kw.get("max_retries", 3)
        self.last_usage: dict = {}

    def complete(self, system: str, user: str, **kw) -> str:
        endpoint = os.getenv("QWEN_ENDPOINT")
        api_key = os.getenv("QWEN_API_KEY")

        if not endpoint:
            raise ValueError("QWEN_ENDPOINT variable is not set")
        if not api_key:
            raise ValueError("QWEN_API_KEY variable is not set")

        payload = {
            "model": kw.get("model", DEFAULT_MODEL),
            "messages": [
                {"role": "system", "content": system + "\n/no_think"},
                {"role": "user", "content": user},
            ],
            "max_tokens": kw.get("max_tokens", 4096),
            "temperature": kw.get("temperature", 0.0),
            "chat_template_kwargs": {"enable_thinking": kw.get("thinking", False)},
        }

        if kw.get("top_p") is not None:
            payload["top_p"] = kw["top_p"]

        if kw.get("stop") is not None:
            payload["stop"] = kw["stop"]

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                self.last_usage = result.get("usage") or {}

                choices = result.get("choices", [])

                if not choices:
                    raise ValueError("Qwen API returned no choices.")

                if choices[0].get("finish_reason") == "length":
                    logging.getLogger("pgia").warning("Qwen reply truncated at max_tokens=%s", payload["max_tokens"])

                message = choices[0].get("message", {})
                content = message.get("content")

                if isinstance(content, str):
                    return content

                if isinstance(content, list):
                    return "".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                    )

                reasoning = message.get("reasoning_content") or message.get("reasoning")
                raise RuntimeError(
                    "Qwen returned no answer content "
                    f"(finish_reason={choices[0].get('finish_reason')}, "
                    f"usage={result.get('usage')}, "
                    f"reasoning_chars={len(reasoning) if isinstance(reasoning, str) else 0}). "
                    "If reasoning_chars is large, thinking used up max_tokens."
                )

            except urllib.error.HTTPError as exc:
                status = exc.code

                if status == 429 or status >= 500:
                    if attempt < self.max_retries:
                        time.sleep(min(2 ** attempt, 8))
                        continue

                error_body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Qwen API HTTP error {status}: {error_body}"
                ) from exc

            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue

                raise RuntimeError(
                    f"Qwen API connection failed after "
                    f"{self.max_retries + 1} attempts: {exc}"
                ) from exc

        raise RuntimeError("Qwen API request failed.")


class _SingletonBackend:
    def __init__(self) -> None:
        self._openai = OpenAIBackend()

    def complete(self, system: str, user: str, *, source_id: str | None = None, **kw) -> str:
        started = time.time()
        text = self._openai.complete(system, user, **kw)
        # record the call (trace). Imported lazily to avoid a hard dependency at import time.
        from logs import trace
        usage = self._openai.last_usage
        trace.log_call(
            system=system,
            system_version=kw.get("system_version"),
            user=user,
            output=text,
            source_id=source_id,
            model=kw.get("model", DEFAULT_MODEL),
            latency_s=round(time.time() - started, 3),
            trace_id=kw.get("trace_id"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        return text


_singleton = _SingletonBackend()


def complete(system: str, user: str, *, source_id: str | None = None, **kw) -> str:
    return _singleton.complete(system, user, source_id=source_id, **kw)
