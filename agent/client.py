"""OpenAI-compatible HTTP client: no implicit retries, no key logging."""
import os
import threading
import time
import requests
from dotenv import load_dotenv

_rate_lock = threading.Lock()
_inflight = threading.BoundedSemaphore(2)
_next_request_at = 0.0


class APIRequestError(RuntimeError):
    def __init__(self, status):
        self.status_code = status
        super().__init__(f"Model API HTTP {status}")


class ChatClient:
    def __init__(self, config):
        load_dotenv()
        self.config = config
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
        self.key = os.getenv("LLM_API_KEY", "")
        if not self.key or self.key == "your_api_key_here":
            raise ValueError("Configure LLM_API_KEY in .env")
        self.session = requests.Session()

    def complete(self, messages, seed):
        global _next_request_at
        with _inflight:
            with _rate_lock:
                time.sleep(max(0, _next_request_at-time.monotonic()))
                _next_request_at = time.monotonic() + self.config.api_interval
            response = self.session.post(self.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self.key},
                json={"model": self.config.model, "messages": messages,
                      "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
                      "seed": seed, "stop": ["Observation:"]}, timeout=self.config.timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                try:
                    cooldown = min(60, max(30, float(response.headers.get("Retry-After", 30))))
                except ValueError:
                    cooldown = 30
                with _rate_lock:
                    _next_request_at = max(_next_request_at, time.monotonic()+cooldown)
            if response.status_code >= 400:
                raise APIRequestError(response.status_code)
        data = response.json()
        choice = data["choices"][0]
        return {"content": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"), "usage": data.get("usage", {}),
                "request_id": data.get("id"), "model": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"), "requested_seed": seed}
