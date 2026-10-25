from tools import search_tool


def test_failed_requests_are_throttled_without_retry(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(search_tool.time, "sleep", sleeps.append)
    monkeypatch.setattr(search_tool.time, "monotonic", lambda: 100)
    monkeypatch.setattr(search_tool, "_next_request_at", 101)
    class Session:
        def get(self, *args, **kwargs):
            calls.append(kwargs)
            class Response:
                status_code = 429
                headers = {"Retry-After": "10"}
            return Response()
    search_tool._get(Session(), "https://example.test", {})
    assert sleeps == [1]
    assert search_tool._next_request_at == 110
    assert len(calls) == 1
