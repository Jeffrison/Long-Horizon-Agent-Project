"""Budgeted ReAct runtime. Gold answers are deliberately absent from this API."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import time

from agent.client import APIClient
from agent.config import AgentConfig
from agent.parser import parse_model_response
from agent.prompts import make_prompt
from tools import build_tools, get_tool_description
from tools.types import ToolResult


def run_agent(query, task_type="QA", max_steps=8, enable_reflection=True, *,
              config=None, client=None, tools=None, allow_python=False):
    if task_type not in {"QA", "MATH"}:
        raise ValueError("task_type must be QA or MATH")
    config = config or AgentConfig(
        mode="react_reflect" if enable_reflection else "react", max_llm_calls=max_steps)
    if tools is None:
        tools = build_tools(task_type, allow_python)
    if config.mode == "direct":
        tools = {}
    elif not tools:
        raise ValueError("No task tool enabled. Math execution needs --allow-python.")
    client = client or APIClient()
    prompt = make_prompt(task_type, get_tool_description(tools), config.mode)
    prefix = [{"role": "system", "content": prompt}, {"role": "user", "content": query}]
    history, trajectory, events = [], [], []
    errors = Counter()
    calls = tool_calls = tool_success = retries = format_streak = 0
    usage_count = prompt_tokens = completion_tokens = 0
    feedback = ""
    start = time.monotonic()

    def finish(reason, answer=None):
        complete_usage = usage_count == calls and calls > 0
        return {
            "status": "success" if reason == "final_answer" else "failed",
            "termination_reason": reason, "error": None if reason == "final_answer" else reason,
            "final_answer": answer, "steps": calls,
            "stats": {"llm_calls": calls, "tool_calls": tool_calls, "tool_successes": tool_success,
                      "retries": retries, "errors": dict(errors),
                      "prompt_tokens": prompt_tokens if complete_usage else None,
                      "completion_tokens": completion_tokens if complete_usage else None,
                      "usage_records_available": usage_count,
                      "elapsed_seconds": round(time.monotonic() - start, 4)},
            "config": config.to_dict(), "events": events, "trajectory": trajectory,
        }

    while calls < config.max_llm_calls:
        messages = deepcopy(prefix + history)
        if feedback:
            messages.append({"role": "user", "content": feedback})
        calls += 1  # Includes all malformed responses and failed provider requests.
        call_start = time.monotonic()
        effective_seed = config.seed + calls - 1
        event = {"kind": "llm", "call": calls, "messages": messages, "effective_seed": effective_seed}
        try:
            response = client.complete(messages, replace(config, seed=effective_seed))
            text = response["text"]
            event.update({"text": text, "usage": response.get("usage", {}),
                          "response_model": response.get("response_model")})
        except Exception as exc:
            # No exception message, endpoint or credentials are written to logs.
            errors["api_error"] += 1
            event.update({"error": type(exc).__name__,
                          "seconds": round(time.monotonic() - call_start, 4)})
            events.append(event)
            return finish("api_error")
        event["seconds"] = round(time.monotonic() - call_start, 4)
        events.append(event)
        trajectory.append({"role": "assistant", "content": text})
        usage = response.get("usage", {})
        if all(type(usage.get(k)) is int for k in ("prompt_tokens", "completion_tokens")):
            usage_count += 1
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
        try:
            parsed = parse_model_response(text, task_type, config.mode)
            event["parsed"] = parsed
            if config.mode == "direct" and parsed["type"] != "finish":
                raise ValueError("Direct mode cannot call tools")
        except ValueError as exc:
            errors["format_error"] += 1
            format_streak += 1
            event["error"] = "format_error"
            if config.mode != "react_reflect" or format_streak > config.max_format_retries:
                return finish("format_error")
            retries += 1
            feedback = "Formatting error: " + str(exc) + ". Return one valid Action or Final Answer."
            # Invalid content is logged, but never put into model history.
            continue
        format_streak = 0
        feedback = ""
        if parsed["type"] == "finish":
            return finish("final_answer", parsed["content"])
        if tool_calls >= config.max_tool_calls:
            errors["tool_budget"] += 1
            return finish("tool_budget")
        registry = {key.lower(): key for key in tools}
        name = registry.get(parsed["action"].lower())
        tool_calls += 1  # Unknown tool attempts also consume the shared tool budget.
        tool_start = time.monotonic()
        if name is None:
            result = ToolResult(False, "Unknown tool. Choose: " + ", ".join(tools), "unknown_tool")
        else:
            try:
                result = tools[name](parsed["action_input"])
                if isinstance(result, str):
                    result = ToolResult(True, result)
                if not isinstance(result, ToolResult):
                    raise TypeError("Tools must return ToolResult")
            except Exception:
                result = ToolResult(False, "Tool raised an exception.", "tool_exception")
        tool_event = {"kind": "tool", "name": name or parsed["action"],
                      "input": parsed["action_input"], "ok": result.ok, "output": result.text,
                      "error": result.error, "seconds": round(time.monotonic() - tool_start, 4)}
        events.append(tool_event)
        observation = "Observation:\n" + result.text[:12000]
        if result.ok:
            tool_success += 1
        else:
            error = result.error or "tool_error"
            errors[error] += 1
            observation += "\nTool error category: " + error
            if config.mode == "react_reflect":
                observation += "\nRevise the action or input using this error; do not repeat it unchanged."
                retries += 1
        pair = [{"role": "assistant", "content": text}, {"role": "user", "content": observation}]
        trajectory.append(pair[-1])
        history = (history + pair)[-2 * config.history_turns:]
        if not result.ok and config.mode != "react_reflect":
            return finish(result.error or "tool_error")
    return finish("llm_budget")
