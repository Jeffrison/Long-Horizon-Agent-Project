"""Budgeted ReAct execution. Labels and scoring never enter this module."""
import time
from dataclasses import asdict, dataclass
from agent.client import ChatClient
from agent.parser import parse_llm_output
from agent.prompts import QA_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT
from tools import TOOL_REGISTRY, get_tool_description
from tools.result import ToolResult


@dataclass(frozen=True)
class AgentConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_calls: int = 8
    max_tokens: int = 600
    temperature: float = 0.1
    history_turns: int = 8
    max_observation_chars: int = 6000
    timeout: float = 60
    seed: int = 0
    retrieval: str = "live"
    context_path: str = "data/contexts/hotpotqa_distractor.jsonl"
    api_interval: float = 1.0

    def __post_init__(self):
        if min(self.max_calls, self.max_tokens, self.history_turns, self.max_observation_chars) < 1:
            raise ValueError("Agent budgets must be positive")
        if self.timeout <= 0 or not 0 <= self.temperature <= 2:
            raise ValueError("Invalid timeout or temperature")
        if self.retrieval not in {"live", "snapshot"}:
            raise ValueError("Unknown retrieval backend")
        if self.api_interval < 0:
            raise ValueError("api_interval must be nonnegative")


def run_agent(query: str, task_type: str = "QA", max_steps: int = 8,
              enable_reflection: bool = True, *, config=None, client=None, tools=None) -> dict:
    """One trajectory; every inference attempt consumes the same call budget.

    Reflection adds explicit feedback on observable protocol/tool failures.
    Both modes can recover. Neither mode inspects answer correctness.
    """
    if task_type not in {"QA", "MATH"}:
        raise ValueError("task_type must be QA or MATH")
    config = config or AgentConfig(max_calls=max_steps)
    client = client or ChatClient(config)
    if tools is None:
        if task_type == "QA" and config.retrieval == "snapshot":
            from eval.data_loader import question_key
            from tools.local_search import load_contexts, LocalSearch
            documents = load_contexts(config.context_path).get(question_key(query))
            if documents is None:
                raise ValueError("Question is not covered by the frozen context snapshot")
            tools = {"Search": LocalSearch(documents)}
        else:
            tools = {name: TOOL_REGISTRY[name] for name in
                     (["Search"] if task_type == "QA" else ["Python_REPL"])}
    template = QA_SYSTEM_PROMPT if task_type == "QA" else MATH_SYSTEM_PROMPT
    system = template.format(tool_descriptions=get_tool_description(tools), tool_names=", ".join(tools))
    if task_type == "QA" and config.retrieval == "snapshot":
        system += "\nSearch retrieves from this question's fixed candidate paragraphs, not live Wikipedia."
    initial = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    history, trajectory, events = [], [], []
    counts = {"llm_calls": 0, "tool_calls": 0, "tool_errors": 0, "parse_errors": 0,
              "reflection_events": 0, "prompt_tokens": 0, "completion_tokens": 0}
    started = time.monotonic()

    def finish(status, answer=None, error=None):
        return {"status": status, "final_answer": answer, "error": error,
                "steps": counts["llm_calls"], **counts, "trajectory": trajectory,
                "events": events, "initial_messages": initial, "config": asdict(config),
                "enable_reflection": enable_reflection,
                "latency_seconds": round(time.monotonic() - started, 4)}

    for call in range(config.max_calls):
        messages = initial + [m for turn in history[-config.history_turns:] for m in turn]
        event = {"call": call + 1, "messages": messages}
        events.append(event)
        counts["llm_calls"] += 1
        call_start = time.monotonic()
        try:
            response = client.complete(messages, config.seed + call)
        except Exception as exc:
            event.update(error="api_error", error_type=type(exc).__name__,
                         http_status=getattr(exc, "status_code", None))
            return finish("failed", error="api_error")
        event.update(response)
        event["latency_seconds"] = round(time.monotonic() - call_start, 4)
        for name in ("prompt_tokens", "completion_tokens"):
            counts[name] += response.get("usage", {}).get(name, 0)
        text = response.get("content") or ""
        assistant = {"role": "assistant", "content": text}
        trajectory.append(assistant)
        try:
            parsed = parse_llm_output(text, task_type)
        except ValueError:
            counts["parse_errors"] += 1
            event["outcome"] = "parse_error"
            feedback = "Observation: Invalid output format."
            if enable_reflection:
                counts["reflection_events"] += 1
                feedback += (" Review the protocol error. Output either Action and Action Input, "
                             "or a nonempty Final Answer. Do not invent an Observation.")
            observation = {"role": "user", "content": feedback}
            trajectory.append(observation)
            history.append([observation])
            continue
        if parsed["type"] == "finish":
            event["outcome"] = "final_answer"
            return finish("success", parsed["content"])
        counts["tool_calls"] += 1
        name = parsed["action"].strip("`*\"'_ ")
        mapping = {key.lower(): key for key in tools}
        canonical = mapping.get(name.lower())
        if canonical is None:
            result = ToolResult(f"Unknown tool: {name}", True, "unknown_tool")
        else:
            try:
                result = tools[canonical](parsed["action_input"])
                if not isinstance(result, ToolResult):
                    raise TypeError("Tools must return ToolResult")
            except Exception as exc:
                result = ToolResult(f"Tool failed: {type(exc).__name__}", True, "tool_exception")
        event["tool"] = {"name": canonical or name, "input": parsed["action_input"], **asdict(result)}
        event["outcome"] = "tool_error" if result.is_error else "tool_success"
        feedback = "Observation:\n" + result.content[:config.max_observation_chars]
        if result.is_error:
            counts["tool_errors"] += 1
            if enable_reflection:
                counts["reflection_events"] += 1
                feedback += ("\nReview the reported error, identify its cause, and revise your "
                             "tool input or choose another action. Do not repeat an identical failed action.")
        observation = {"role": "user", "content": feedback}
        trajectory.append(observation)
        history.append([assistant, observation])
    return finish("failed", error="max_calls_reached")
