import pytest
from eval.metrics import qa_match, qa_f1, math_match, parse_latex_to_float


@pytest.mark.parametrize("pred,gold", [("Both are American", "yes"), ("not sure", "no"),
                                     ("Yorkshire", "York"), ("yes or no", "yes"), ("", "")])
def test_no_lenient_qa_false_positives(pred, gold):
    assert not qa_match(pred, gold)


def test_qa_metrics():
    assert qa_match("The United States.", "United States")
    assert qa_f1("John Smith", "John") == pytest.approx(2/3)
    assert qa_f1("yes indeed", "yes") == 0


@pytest.mark.parametrize("pred,gold", [
    ("1/576", r"\frac{1}{576}"), (r"\sqrt{2}", "1.4142135623730951"),
    (r"\boxed{70}", "70"), (r"\frac{1}{\sqrt{4}}", "0.5"),
    ("1,000", "1000"), ("-2^3", "-8"), ("0", "0"), ("5!", "120"),
    (r"2\sqrt{2}", "2.82842712474619")])
def test_math_equivalence(pred, gold):
    assert math_match(pred, gold)


@pytest.mark.parametrize("pred,gold", [("1/2", "1"), ("Answer is 42", "42"),
    ("0.0001", "0"), ("NaN", "0"), ("1,2", "12"), ("42 or 43", "42"), ("", "")])
def test_math_rejects_first_number_and_loose_tolerance(pred, gold):
    assert not math_match(pred, gold)


@pytest.mark.parametrize("text", ["__import__('os').getcwd()", "10**100000000", "1/0", "sqrt(-1)", "[1][0]"])
def test_math_parser_bounded_and_not_eval(text):
    assert parse_latex_to_float(text) is None
