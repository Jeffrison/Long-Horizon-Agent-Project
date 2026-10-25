"""OpenAI-compatible HTTP client: no implicit retries, no key logging."""
import os
import requests
from dotenv import load_dotenv


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
        response = self.session.post(self.base_url + "/chat/completions",
            headers={"Authorization": "Bearer " + self.key},
            json={"model": self.config.model, "messages": messages,
                  "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
                  "seed": seed, "stop": ["Observation:"]}, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        return {"content": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"), "usage": data.get("usage", {}),
                "request_id": data.get("id"), "model": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"), "requested_seed": seed}
