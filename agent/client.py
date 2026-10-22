"""API adapter. Offline imports never read credentials or send requests."""
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_local_env(path=None):
    path = Path(path or Path(__file__).resolve().parents[1] / ".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class APIClient:
    def __init__(self, base_url=None, api_key=None, timeout=60):
        load_local_env()
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.timeout = timeout
        if not self.base_url or not self.api_key:
            raise ValueError("Set LLM_BASE_URL and LLM_API_KEY in the local .env")

    def complete(self, messages, config):
        payload = {"model": config.model, "messages": messages,
                   "temperature": config.temperature, "max_tokens": config.max_new_tokens,
                   "seed": config.seed, "stop": ["Observation:"]}
        request = Request(self.base_url + "/chat/completions",
                          data=json.dumps(payload).encode("utf-8"),
                          headers={"Authorization": "Bearer " + self.api_key,
                                   "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"provider_http_{exc.code}") from None
        except (URLError, TimeoutError):
            raise RuntimeError("provider_connection_error") from None
        try:
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError
            return {"text": text.strip(), "usage": data.get("usage") or {},
                    "response_model": data.get("model"), "response_id": data.get("id")}
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("provider_invalid_response") from None


class ScriptedClient:
    """Deterministic regression fixture, never a model benchmark."""
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, config):
        self.requests.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {"text": response, "usage": {}}
