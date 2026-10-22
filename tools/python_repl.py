"""Opt-in local math execution. A child process is NOT a security sandbox."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from tools.types import ToolResult

ALLOWED_IMPORTS = {"math", "sympy", "numpy", "scipy", "itertools", "fractions", "statistics", "collections", "decimal"}
BLOCKED_CALLS = {"open", "exec", "eval", "compile", "input", "getattr", "setattr", "delattr", "globals", "locals", "vars", "breakpoint"}


def auto_inject_print(code):
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        value = tree.body[-1].value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "print"):
            # Replace: appending would execute side-effecting calls twice.
            tree.body[-1] = ast.Expr(ast.Call(ast.Name("print", ast.Load()), [value], []))
    return ast.unparse(ast.fix_missing_locations(tree))


def validate_code(code):
    if len(code) > 12000:
        raise ValueError("Code too long")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] not in ALLOWED_IMPORTS for a in node.names):
                raise ValueError("Only math-related imports are allowed")
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS):
            raise ValueError("Only math-related imports are allowed")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in BLOCKED_CALLS):
            raise ValueError("Disallowed name")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in {"load", "save", "loadtxt", "savetxt", "tofile", "fromfile", "dump", "dumps", "system"}):
            raise ValueError("Disallowed attribute")


def run_python(code, timeout=10):
    code = code.strip()
    fence = chr(96) * 3
    if code.startswith(fence):
        code = "\n".join(code.splitlines()[1:]).removesuffix(fence).strip()
    try:
        validate_code(code)
        code = auto_inject_print(code)
    except (SyntaxError, ValueError) as exc:
        return ToolResult(False, f"Invalid math code: {exc}", "invalid_code")
    # Defense in depth only. Production execution needs a real OS/container sandbox.
    env = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "SYSTEMDRIVE"}}
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        with tempfile.TemporaryDirectory(prefix="agent_math_") as directory:
            script = Path(directory) / "task.py"
            script.write_text(code, encoding="utf-8")
            with (Path(directory) / "stdout.txt").open("w+b") as stdout, (Path(directory) / "stderr.txt").open("w+b") as stderr:
                result = subprocess.run([sys.executable, "-I", str(script)], cwd=directory,
                                        stdout=stdout, stderr=stderr, timeout=timeout, env=env)
                stdout.seek(0)
                stderr.seek(0)
                text = stdout.read(8000).decode("utf-8", errors="replace").strip()
                error = stderr.read(4000).decode("utf-8", errors="replace").strip()
            if result.returncode:
                return ToolResult(False, error or "Python exited with an error.", "execution_error")
            if not text:
                return ToolResult(False, "No output. Print the result.", "no_output")
            return ToolResult(True, text)
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"Execution exceeded {timeout} seconds.", "timeout")
    except OSError:
        return ToolResult(False, "Could not start the local Python process.", "process_error")


def execute_python(code, timeout=10):
    return run_python(code, timeout).text
