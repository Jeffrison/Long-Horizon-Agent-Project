import subprocess
import tempfile
import os
import re
import ast

def auto_inject_print(code: str) -> str:
    """
    沙箱级代码劫持处理器：
    1. 重写内置的 print 函数，拦截 sympy 符号对象并强制转为 Float。
    2. AST 解析：如果代码最后一行是一个孤立的表达式（忘记写 print），自动为其包裹 print()。
    """
    
    # 1：重写 builtins.print
    patch_code = """
import builtins
_orig_print = builtins.print

def _smart_print(*args, **kwargs):
    new_args =[]
    for a in args:
        # 如果对象的模块名以 sympy 开头，尝试调用 .evalf() 并转为 float
        if type(a).__module__.startswith('sympy'):
            try:
                a = float(a.evalf())
            except Exception:
                pass
        new_args.append(a)
    _orig_print(*new_args, **kwargs)

builtins.print = _smart_print
"""
    
    # 2：AST 静态分析
    try:
        tree = ast.parse(code)
        # 如果代码不为空，且最后一条语句是“表达式（Expr）”而不是“赋值等语句”
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body[-1].value
            
            # 检查它是不是已经在调用 print 了
            is_print_call = isinstance(last_expr, ast.Call) and getattr(last_expr.func, 'id', '') == 'print'
            
            if not is_print_call:
                # 提取最后一行，强行套上 print()
                last_line_str = ast.unparse(last_expr)
                code += f"\nprint({last_line_str})"
    except Exception:
        # 如果代码本身有 SyntaxError，直接忽略，让底层的 python 子进程去真实报错
        pass

    # 将劫持补丁注入到大模型代码的最上方
    return patch_code + "\n" + code


def execute_python(code: str, timeout: int = 10) -> str:
    """
    在一个隔离的子进程中执行 Python 代码。
    """
    # 1. 暴力抹除 Markdown 标记
    code = re.sub(r"^```[a-zA-Z]*\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"^```\n?", "", code, flags=re.MULTILINE)
    code = code.strip()

    # 2. 在执行前，拦截并自动注入 print 与类型转换
    code = auto_inject_print(code)

    # 3. 隔离执行
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(code)
        
        result = subprocess.run(
            ['python', path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = ""
        if result.stdout:
            output += result.stdout
            
        if result.stderr:
            output += f"\n[Error Traceback]:\n{result.stderr}"
            
        if not output.strip():
            return "Observation: Code executed successfully, but there is no output. Please remember to output the results you want to see!"
        
        return output.strip()

    except subprocess.TimeoutExpired:
        return f"Observation: Code execution timed out (exceeded {timeout} seconds). You might have written an infinite loop."
    except Exception as e:
        return f"Observation: Execution environment system error: {str(e)}"
    finally:
        if os.path.exists(path):
            os.remove(path)