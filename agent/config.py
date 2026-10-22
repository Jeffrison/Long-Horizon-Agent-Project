"""Serializable, explicit budgets shared by every evaluation mode."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    mode: str = "react_reflect"
    max_llm_calls: int = 8
    max_tool_calls: int = 6
    max_new_tokens: int = 600
    history_turns: int = 2
    max_format_retries: int = 3
    temperature: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if self.mode not in {"direct", "react", "react_reflect"}:
            raise ValueError("mode must be direct, react, or react_reflect")
        for key in ("max_llm_calls", "max_new_tokens", "history_turns"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.max_tool_calls < 0 or self.max_format_retries < 0:
            raise ValueError("tool/retry budgets must be nonnegative")

    def to_dict(self):
        return asdict(self)
