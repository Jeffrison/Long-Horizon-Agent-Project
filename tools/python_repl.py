"""Restricted subprocess runner; NOT a security sandbox for hostile code."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from tools.result import ToolResult

ALLOWED_IMPORTS = {"math", "sympy", "numpy", "scipy", "statistics", "itertools", "functools",
                   "collections", "decimal", "fractions", "random", "warnings"}
BLOCKED_NAMES = {"open", "eval", "exec", "compile", "globals", "locals", "vars", "getattr",
                 "setattr", "delattr", "__import__", "input", "breakpoint", "help"}
BLOCKED_ATTRIBUTES = {"load", "save", "savez", "loadtxt", "savetxt", "fromfile", "tofile",
                      "memmap", "ctypeslib", "f2py", "read", "write", "dump", "dumps", "loads"}


def auto_inject_print(code: str) -> str:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(m.split(".")[0] not in ALLOWED_IMPORTS for m in modules):
                raise ValueError("Import is outside the math library allowlist")
        if isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            raise ValueError("Disallowed builtin")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BLOCKED_ATTRIBUTES):
            raise ValueError("Disallowed attribute")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        value = tree.body[-1].value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "print"):
            # Replace, rather than append: expressions with side effects run once.
            tree.body[-1].value = ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[value], keywords=[])
    return ast.unparse(ast.fix_missing_locations(tree))


def execute_python(code: str, timeout: float = 10) -> ToolResult:
    try:
        code = auto_inject_print(code)
    except (SyntaxError, ValueError) as exc:
        return ToolResult(f"{type(exc).__name__}: {exc}", True, "invalid_code")
    env = {k: v for k, v in os.environ.items() if k.upper() in
           {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "SYSTEMDRIVE"}}
    env.update(PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="agent_math_") as directory:
        path = Path(directory) / "program.py"
        path.write_text(code, encoding="utf-8")
        with open(Path(directory) / "stdout", "w+b") as stdout, open(Path(directory) / "stderr", "w+b") as stderr:
            try:
                process = subprocess.run([sys.executable, "-I", "-X", "utf8", str(path)],
                    cwd=directory, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
            except subprocess.TimeoutExpired:
                return ToolResult(f"Execution timed out after {timeout}s", True, "timeout")
            except OSError:
                return ToolResult("Execution environment unavailable", True, "environment_error")
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read(12000).decode("utf-8", "replace")
            err = stderr.read(4000).decode("utf-8", "replace")
        if process.returncode:
            return ToolResult((out + "\n" + err).strip(), True, "execution_error")
        return ToolResult(out.strip() or "Code completed without output.")
