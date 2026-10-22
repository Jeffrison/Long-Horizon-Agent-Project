"""Answer-only HotpotQA EM/F1 and conservative numeric math matching.
Convention: https://github.com/hotpotqa/hotpot
This project does not evaluate supporting facts or report joint scores.
"""
import ast
from collections import Counter
import math
import operator
import re
import string


def normalize_text(value):
    text = str(value if value is not None else "").lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_scores(prediction, ground_truth):
    pred, gold = normalize_text(prediction), normalize_text(ground_truth)
    em = float(pred == gold)
    if (pred in {"yes", "no", "noanswer"} or gold in {"yes", "no", "noanswer"}) and pred != gold:
        return {"em": em, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    p, g = pred.split(), gold.split()
    common = sum((Counter(p) & Counter(g)).values())
    precision = common / len(p) if p else 0.0
    recall = common / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if common else 0.0
    return {"em": em, "f1": f1, "precision": precision, "recall": recall}


def qa_match(prediction, ground_truth):
    return bool(qa_scores(prediction, ground_truth)["em"])


def parse_latex_to_float(value):
    """Finite scalar arithmetic only. Unsupported symbolic answers return None."""
    if value is None:
        return None
    text = str(value).strip().strip("$").replace(" ", "")
    if len(text) > 512:
        return None
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\pi", "pi")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    for _ in range(32):
        before = text
        text = re.sub(r"\\(?:boxed|text){([^{}]+)}", r"(\1)", text)
        text = re.sub(r"\\frac{([^{}]+)}{([^{}]+)}", r"((\1)/(\2))", text)
        text = re.sub(r"\\sqrt{([^{}]+)}", r"sqrt(\1)", text)
        text = re.sub(r"\^{([^{}]+)}", r"**(\1)", text)
        if text == before:
            break
    text = re.sub(r"(\d+)!", r"factorial(\1)", text).replace("^", "**")
    text = re.sub(r"(?<=\d)(?=sqrt\(|pi\b)", "*", text)
    binary = {ast.Add: operator.add, ast.Sub: operator.sub,
              ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (abs(b) > 12 or abs(a) > 1e12):
                raise ValueError("power limit")
            return binary[type(node.op)](a, b)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1 and not node.keywords:
            arg = visit(node.args[0])
            if node.func.id == "sqrt":
                return math.sqrt(arg)
            if node.func.id == "factorial" and arg.is_integer() and 0 <= arg <= 100:
                return float(math.factorial(int(arg)))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(text, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        result = float(visit(tree.body))
        return result if math.isfinite(result) else None
    except (ValueError, SyntaxError, TypeError, OverflowError, ZeroDivisionError):
        return None


def math_match(prediction, ground_truth, abs_tol=1e-6, rel_tol=1e-6):
    pred, gold = parse_latex_to_float(prediction), parse_latex_to_float(ground_truth)
    if pred is not None and gold is not None:
        return math.isclose(pred, gold, abs_tol=abs_tol, rel_tol=rel_tol)
    return bool(str(prediction or "").strip()) and str(prediction).strip() == str(ground_truth).strip()
