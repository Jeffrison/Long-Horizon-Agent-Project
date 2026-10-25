import ast
import pytest
from agent.parser import parse_llm_output
from tools.python_repl import auto_inject_print, execute_python


def test_preserve_symbolic_answer_and_code_quotes():
    assert parse_llm_output("Final Answer: (1, 2)")["content"] == "(1, 2)"
    assert parse_llm_output("Action: Python_REPL\nAction Input:\n```python\n'abc'\n```")["action_input"] == "'abc'"


@pytest.mark.parametrize("text", ["Final Answer:", "Action: T\nAction Input:",
    "Action: T\nAction Input: x\nFinal Answer: 1", "Final Answer: 1\nFinal Answer: 2",
    "Action: Python_REPL\nAction Input:\n```python\nx=1"])
def test_bad_protocol(text):
    with pytest.raises(ValueError):
        parse_llm_output(text)


def test_code_fence_cannot_forge_final_answer():
    parsed = parse_llm_output('Action: Python_REPL\nAction Input:\n```python\nprint("Final Answer: 42")\n```')
    assert parsed["type"] == "action"


def test_auto_print_does_not_duplicate_side_effects():
    tree = ast.parse(auto_inject_print("x=[1,2]\nx.pop()"))
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "pop"
               for n in ast.walk(tree)) == 1
    assert execute_python("x=[1,2]\nx.pop()").content == "2"


def test_real_python_errors_timeout_and_symbolic_output():
    assert execute_python("1/0").error_type == "execution_error"
    assert execute_python("while True: pass", timeout=0.2).error_type == "timeout"
    assert execute_python("import sympy as s\ns.Rational(1,576)").content == "1/576"
    assert execute_python("import os\nos.environ").is_error


def test_nonerror_stderr_is_not_misclassified():
    assert not execute_python("import warnings\nwarnings.warn('notice')\n42").is_error
