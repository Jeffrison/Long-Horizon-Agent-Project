"""Conservative answer-only metrics; no substring/oracle matching."""
import ast
from collections import Counter
import math
import operator
import re
import string


def normalize_text(value) -> str:
    value = str(value or "").lower()
    value = "".join(c for c in value if c not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def qa_match(prediction, ground_truth) -> bool:
    pred, gold = normalize_text(prediction), normalize_text(ground_truth)
    return bool(pred and gold and pred == gold)


def qa_f1(prediction, ground_truth) -> float:
    pred, gold = normalize_text(prediction), normalize_text(ground_truth)
    if not pred or not gold:
        return 0.0
    if pred in {"yes", "no", "noanswer"} or gold in {"yes", "no", "noanswer"}:
        return float(pred == gold)
    common = sum((Counter(pred.split()) & Counter(gold.split())).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred.split()), common / len(gold.split())
    return 2 * precision * recall / (precision + recall)


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}
FUNCTIONS = {"sqrt": math.sqrt, "factorial": math.factorial}


def _number(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if abs(node.value) > 1e100:
            raise ValueError("Constant too large")
        return node.value
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return -_number(node.operand) if isinstance(node.op, ast.USub) else _number(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = _number(node.left), _number(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e10):
            raise ValueError("Power too large")
        result = OPS[type(node.op)](a, b)
        if isinstance(result, complex) or not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("Nonfinite result")
        return result
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
        if len(node.args) != 1 or node.keywords:
            raise ValueError("Invalid call")
        value = _number(node.args[0])
        if node.func.id == "factorial":
            if not 0 <= value <= 100 or int(value) != value:
                raise ValueError("Invalid factorial")
            value = int(value)
        return FUNCTIONS[node.func.id](value)
    raise ValueError("Unsupported mathematical syntax")


def _unwrap_box(s):
    if not s.startswith(r"\boxed{") or not s.endswith("}"):
        return s
    depth = 0
    for index, char in enumerate(s[6:], 6):
        depth += (char == "{") - (char == "}")
        if depth == 0:
            return s[7:-1] if index == len(s) - 1 else s
    return s


def parse_latex_to_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if len(s) > 512:
        return None
    s = _unwrap_box(s.strip("$").strip())
    s = s.replace(r"\left", "").replace(r"\right", "").replace(r"\,", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    # Allow well-formed numeric thousands separators, never strip arbitrary commas.
    s = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?", lambda m: m.group().replace(",", ""), s)
    if "," in s:
        return None
    s = s.replace(" ", "").replace(r"\pi", "pi").replace(r"\cdot", "*").replace(r"\times", "*")
    for _ in range(16):
        previous = s
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
        if s == previous:
            break
    s = s.replace("^", "**")
    s = re.sub(r"(\d+)!", r"factorial(\1)", s)
    s = re.sub(r"(\d|\))(sqrt|pi|\()", r"\1*\2", s)
    try:
        if "{" in s or "}" in s:
            return None
        tree = ast.parse(s, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        result = float(_number(tree.body))
        return result if math.isfinite(result) else None
    except (ValueError, SyntaxError, OverflowError, ZeroDivisionError, TypeError, RecursionError):
        return None


def math_match(prediction, ground_truth, rel_tol=1e-6, abs_tol=1e-8) -> bool:
    pred, gold = parse_latex_to_float(prediction), parse_latex_to_float(ground_truth)
    if pred is not None and gold is not None:
        return math.isclose(pred, gold, rel_tol=rel_tol, abs_tol=abs_tol)
    # Conservative exact expression fallback; do not pick the first number from prose.
    if prediction is None or ground_truth is None:
        return False
    norm = lambda x: re.sub(r"\s+", "", str(x).strip().strip("$"))
    return bool(norm(prediction) and norm(prediction) == norm(ground_truth))
