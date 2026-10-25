"""Exact rational arithmetic for integer answers; conservative relative numeric fallback."""
import ast
from fractions import Fraction
import math
import operator
import re

_OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow}


def _limit(value):
    if isinstance(value,Fraction):
        if max(value.numerator.bit_length(),value.denominator.bit_length())>4096 or abs(value)>10**100:
            raise ValueError("Expression exceeds exact arithmetic bounds")
    elif not math.isfinite(value) or abs(value)>1e100:
        raise ValueError("Expression exceeds numeric bounds")
    return value


def _evaluate(node,source):
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):
        if not math.isfinite(node.value) or abs(node.value)>1e100:
            raise ValueError("Invalid constant")
        raw=ast.get_source_segment(source,node)
        if type(node.value) is int:
            return _limit(Fraction(node.value))
        exponent=re.search(r"[eE]([+-]?\d+)",raw)
        if exponent and abs(int(exponent.group(1)))>400:
            raise ValueError("Exponent too large")
        return _limit(Fraction(raw))
    if isinstance(node,ast.Name) and node.id=="pi":
        return math.pi
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        value=_evaluate(node.operand,source)
        return -value if isinstance(node.op,ast.USub) else value
    if isinstance(node,ast.BinOp) and type(node.op) in _OPS:
        left,right=_evaluate(node.left,source),_evaluate(node.right,source)
        if isinstance(node.op,ast.Pow):
            if abs(right)>100:
                raise ValueError("Power too large")
            if isinstance(right,Fraction):
                right=int(right) if right.denominator==1 else float(right)
        value=_OPS[type(node.op)](left,right)
        if isinstance(value,complex):
            raise ValueError("Non-real result")
        return _limit(value)
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and len(node.args)==1 and not node.keywords:
        value=_evaluate(node.args[0],source)
        if node.func.id=="factorial":
            if not 0<=value<=100 or int(value)!=value:
                raise ValueError("Invalid factorial")
            return _limit(Fraction(math.factorial(int(value))))
        if node.func.id=="sqrt":
            if value<0:
                raise ValueError("Negative square root")
            if isinstance(value,Fraction):
                numerator,denominator=math.isqrt(value.numerator),math.isqrt(value.denominator)
                if numerator*numerator==value.numerator and denominator*denominator==value.denominator:
                    return Fraction(numerator,denominator)
            return _limit(math.sqrt(value))
    raise ValueError("Unsupported numeric syntax")


def parse_exact_number(value):
    if value is None:
        return None
    s=str(value).strip().strip("$").strip()
    if len(s)>512:
        return None
    if s.startswith(r"\boxed{") and s.endswith("}"):
        s=s[7:-1]
    s=s.replace(r"\left","").replace(r"\right","").replace(r"\,","")
    s=s.replace(r"\dfrac",r"\frac").replace(r"\tfrac",r"\frac")
    s=re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?",lambda m:m.group().replace(",",""),s)
    if "," in s:
        return None
    s=s.replace(" ","").replace(r"\pi","pi").replace(r"\cdot","*").replace(r"\times","*")
    for _ in range(16):
        old=s
        s=re.sub(r"\\sqrt\{([^{}]+)\}",r"sqrt(\1)",s)
        s=re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}",r"((\1)/(\2))",s)
        s=re.sub(r"\^\{([^{}]+)\}",r"**(\1)",s)
        if s==old:
            break
    s=s.replace("^","**")
    s=re.sub(r"(\d+)!",r"factorial(\1)",s)
    s=re.sub(r"(\d|\))(sqrt|pi|\()",r"\1*\2",s)
    try:
        tree=ast.parse(s,mode="eval")
        if sum(1 for _ in ast.walk(tree))>128:
            return None
        return _evaluate(tree.body,s)
    except (ValueError,SyntaxError,OverflowError,ZeroDivisionError,TypeError,RecursionError):
        return None


def exact_math_match(prediction,ground_truth,rel_tol=1e-6):
    pred,gold=parse_exact_number(prediction),parse_exact_number(ground_truth)
    if gold is not None and pred is not None:
        if isinstance(gold,Fraction) and gold.denominator==1:
            # A decimal approximation or off-by-one large integer cannot pass by relative tolerance.
            return isinstance(pred,Fraction) and pred==gold
        return math.isclose(float(pred),float(gold),rel_tol=rel_tol,abs_tol=0)
    if prediction is None or ground_truth is None:
        return False
    normalize=lambda value:re.sub(r"\s+","",str(value).strip().strip("$"))
    return bool(normalize(prediction) and normalize(prediction)==normalize(ground_truth))

