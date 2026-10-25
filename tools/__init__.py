from .search_tool import search
from .python_repl import execute_python

# 构建工具注册表，键为工具名称，值为对应的执行函数
TOOL_REGISTRY = {
    "Search": search,
    "Python_REPL": execute_python
}

def get_tool_description(selected=None) -> str:
    """
    后续用于放入 Prompt 中，告诉模型有哪些工具可用
    """
    descriptions = {
        "Search": "Search: Wikipedia retrieval. Input one exact entity name; returns titles and excerpts.",
        "Python_REPL": "Python_REPL: Fresh restricted Python subprocess. Input self-contained math code; returns output or a structured error.",
    }
    return "\n".join(descriptions.get(name, name) for name in (TOOL_REGISTRY if selected is None else selected))
