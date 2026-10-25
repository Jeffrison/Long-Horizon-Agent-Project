"""HTTPS Wikipedia retrieval with bounded requests and a shared disk cache."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import requests
from tools.result import ToolResult

_lock = threading.Lock()
_next_request_at = 0.0


def _get(session, endpoint, params):
    """Throttle every HTTP request, including failed ones; never retry implicitly."""
    global _next_request_at
    time.sleep(max(0, _next_request_at - time.monotonic()))
    try:
        response = session.get(endpoint, params=params, timeout=15)
        delay = 1.5
        if response.status_code == 429:
            try:
                delay = max(delay, min(60, float(response.headers.get("Retry-After", "5"))))
            except ValueError:
                delay = 5
        _next_request_at = time.monotonic() + delay
        return response
    except requests.RequestException:
        _next_request_at = time.monotonic() + 1.5
        raise


def search(query: str, max_results: int = 2) -> ToolResult:
    query = query.strip().strip("\"'")
    if not query:
        return ToolResult("Empty search query", True, "invalid_input")
    key = hashlib.sha256(f"wikipedia-v2:{max_results}:{query.casefold()}".encode()).hexdigest()
    cache = Path(os.getenv("SEARCH_CACHE_DIR", ".cache/wikipedia")) / (key + ".json")
    with _lock:
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            return ToolResult(data["content"], cached=True)
        if os.getenv("SEARCH_CACHE_ONLY") == "1":
            return ToolResult("Search cache miss in offline mode", True, "cache_miss")
        try:
            session = requests.Session()
            session.headers["User-Agent"] = "LongHorizonAgent/2.0 (research evaluation; github.com/Jeffrison/Long-Horizon-Agent-Project)"
            endpoint = "https://en.wikipedia.org/w/api.php"
            data = _get(session, endpoint, {"action": "query", "list": "search", "srsearch": query,
                "srlimit": max_results, "format": "json"})
            data.raise_for_status()
            payload = data.json()
            if "error" in payload:
                return ToolResult("Wikipedia API returned an error", True, "retrieval_api_error")
            titles = [r["title"] for r in payload.get("query", {}).get("search", [])]
            if not titles:
                content = "No results found."
            else:
                response = _get(session, endpoint, {"action": "query", "prop": "extracts",
                    "exintro": 1, "explaintext": 1, "titles": "|".join(titles), "format": "json"})
                response.raise_for_status()
                pages = response.json().get("query", {}).get("pages", {}).values()
                content = "\n\n".join(f"Title: {p.get('title')}\nSnippet: {p.get('extract', '')[:2400]}"
                                        for p in sorted(pages, key=lambda p: p.get("title", "")))
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps({"query": query, "content": content}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(cache)
            return ToolResult(content)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            return ToolResult(f"Wikipedia HTTP {status}", True, f"retrieval_http_{status}")
        except requests.Timeout:
            return ToolResult("Wikipedia request timed out", True, "retrieval_timeout")
        except (requests.RequestException, ValueError, KeyError):
            return ToolResult("Wikipedia request failed or returned invalid JSON", True, "retrieval_error")
