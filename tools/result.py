from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False
    error_type: str | None = None
    cached: bool = False
