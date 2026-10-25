from agent.react_loop import AgentConfig, run_agent
from tools.result import ToolResult


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, seed):
        self.requests.append((messages, seed))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return {"content": value, "finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_no_ground_truth_entrypoint():
    import inspect
    assert "ground_truth" not in inspect.signature(run_agent).parameters
    source = inspect.getsource(run_agent)
    assert "math_match" not in source and "qa_match" not in source


def test_all_parse_attempts_use_budget_in_both_modes():
    for enabled in [False, True]:
        client = ScriptedClient(["bad"] * 3)
        result = run_agent("query", config=AgentConfig(max_calls=3), client=client, tools={},
                           enable_reflection=enabled)
        assert result["llm_calls"] == result["parse_errors"] == 3
        assert result["error"] == "max_calls_reached"
        assert result["prompt_tokens"] == 30
        assert result["reflection_events"] == (3 if enabled else 0)
        assert len(client.requests) == 3


def test_both_modes_can_recover_from_tool_errors():
    for enabled in [False, True]:
        client = ScriptedClient(["Action: Python_REPL\nAction Input: x", "Final Answer: 42"])
        result = run_agent("query", "MATH", config=AgentConfig(max_calls=2), client=client,
                           tools={"Python_REPL": lambda _: ToolResult("problem", True, "execution_error")},
                           enable_reflection=enabled)
        assert result["final_answer"] == "42" and result["llm_calls"] == 2
        assert result["tool_errors"] == 1


def test_correct_tool_output_does_not_finish_automatically():
    result = run_agent("query", "MATH", config=AgentConfig(max_calls=1),
        client=ScriptedClient(["Action: Python_REPL\nAction Input: 42"]),
        tools={"Python_REPL": lambda _: ToolResult("42")})
    assert result["status"] == "failed" and result["final_answer"] is None


def test_api_error_is_recorded_without_retry_or_secrets():
    result = run_agent("query", client=ScriptedClient([RuntimeError("secret-key")]), tools={})
    assert result["llm_calls"] == 1 and result["error"] == "api_error"
    assert "secret-key" not in str(result)


def test_unknown_tool_is_structured_error():
    result = run_agent("query", config=AgentConfig(max_calls=1),
                       client=ScriptedClient(["Action: Unknown\nAction Input: x"]), tools={})
    assert result["tool_errors"] == 1
    assert result["events"][0]["tool"]["error_type"] == "unknown_tool"


def test_window_keeps_complete_tool_turns():
    client = ScriptedClient(["Action: T\nAction Input: 1", "Action: T\nAction Input: 2", "Final Answer: 2"])
    run_agent("query", config=AgentConfig(history_turns=1), client=client, tools={"T": lambda s: ToolResult(s)})
    last = client.requests[-1][0]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "user"]
    assert "Input: 1" not in str(last)
