"""Versioned independent graders; strict-v2 remains available for historical audits."""
from eval.metrics import qa_match, math_match_v2
from eval.exact_math import exact_math_match

VERSIONS=("strict-v2","strict-v3")


def score_answer(prediction,ground_truth,task,version="strict-v3"):
    if version not in VERSIONS or task not in {"QA","MATH"}:
        raise ValueError("Unknown scoring version or task")
    if task=="QA":
        return qa_match(prediction,ground_truth)
    return (math_match_v2 if version=="strict-v2" else exact_math_match)(prediction,ground_truth)
