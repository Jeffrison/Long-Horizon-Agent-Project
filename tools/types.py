from dataclasses import dataclass


@dataclass
class ToolResult:
    ok: bool
    text: str
    error: str | None = None
