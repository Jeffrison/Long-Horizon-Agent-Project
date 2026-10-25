from types import SimpleNamespace
import pytest
from agent import client as module


def test_http_status_recorded_without_retry(monkeypatch):
    from agent.react_loop import AgentConfig
    monkeypatch.setenv("LLM_API_KEY", "unit-test-key")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    monkeypatch.setattr(module, "_next_request_at", 0)
    client = module.ChatClient(AgentConfig())
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(status_code=429, headers={"Retry-After":"40"})
    monkeypatch.setattr(client.session,"post",post)
    with pytest.raises(module.APIRequestError) as exc:
        client.complete([],11)
    assert exc.value.status_code == 429
    assert module._next_request_at == 140
    assert len(calls) == 1
