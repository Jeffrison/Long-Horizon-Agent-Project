"""Task-specific tools. Python execution requires explicit opt-in."""
from tools.search_tool import WikipediaSearch
from tools.python_repl import run_python


def build_tools(task_type, allow_python=False):
    if task_type == "QA":
        return {"Search": WikipediaSearch()}
    if task_type == "MATH" and allow_python:
        return {"Python_REPL": run_python}
    return {}


def get_tool_description(tools=None):
    descriptions = {"Search": "Search English Wikipedia for an entity; returns evidence snippets.",
                    "Python_REPL": "Execute short Python math code; returns printed output or errors."}
    return "\n".join(f"{name}: {descriptions.get(name, 'External tool')}" for name in (tools or {}))
