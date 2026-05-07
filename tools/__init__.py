from .search_tool import search
from .python_repl import execute_python

# 构建工具注册表，键为工具名称，值为对应的执行函数
TOOL_REGISTRY = {
    "Search": search,
    "Python_REPL": execute_python
}

def get_tool_description() -> str:
    """
    后续用于放入 Prompt 中，告诉模型有哪些工具可用
    """
    return """
1. Search: Information retrieval tool. Input a search query string, returns relevant web snippets.
2. Python_REPL: Python code execution environment. Input your valid Python code string, it will execute and return the stdout or error traceback.
"""