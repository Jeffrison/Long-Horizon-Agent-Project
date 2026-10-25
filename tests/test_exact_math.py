from fractions import Fraction
import pytest
from eval.exact_math import exact_math_match, parse_exact_number


def test_large_integer_precision_and_exact_factorial():
    assert not exact_math_match("9007199254740992","9007199254740993")
    assert not exact_math_match("1000001","1000000")
    assert not exact_math_match("1e34",r"2^{25} \cdot 26!")
    assert exact_math_match("2**25*factorial(26)",r"2^{25} \cdot 26!")


def test_rationals_zero_and_decimals():
    assert parse_exact_number("0.1+0.2")==Fraction(3,10)
    assert exact_math_match("0.5",r"\frac{1}{2}")
    assert exact_math_match("0.001736111111111111",r"\frac{1}{576}")
    assert not exact_math_match("0.000000001","0")
    assert not exact_math_match("0",r"\frac{1}{100000000000}")
    assert exact_math_match(r"\sqrt{4}","2")
    assert not exact_math_match("0","1e-400")
    assert exact_math_match("1e-400","1e-400")


@pytest.mark.parametrize("text",["1e-999999999999","10**1000000","__import__('os')","sqrt(-1)"])
def test_exact_parser_bounds(text):
    assert parse_exact_number(text) is None


def test_invalid_numeric_answers_do_not_match_themselves():
    for text in ["NaN", "inf", "1/0", "sqrt(-1)", "x+1"]:
        assert not exact_math_match(text,text)
