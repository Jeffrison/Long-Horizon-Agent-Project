from collections import Counter
from eval.analyze_failures import primary_reason, flags


def test_primary_taxonomy_partitions_all_outcomes():
    base = {"task":"MATH","is_correct":False,"result":{"status":"success","final_answer":"1/2","events":[]}}
    correct = {**base,"is_correct":True}
    timeout = {**base,"result":{"error":"api_error","events":[{"error_type":"ReadTimeout"}]}}
    outcomes = Counter(primary_reason(row) for row in [base,correct,timeout])
    assert outcomes == {"wrong_numeric_final":1,"correct":1,"api_timeout":1}
    assert sum(outcomes.values()) == 3


def test_recovery_flag_is_not_terminal_success_alone():
    row = {"is_correct":False,"result":{"status":"success","tool_errors":1,"events":[]}}
    assert not flags(row)["error_then_correct"]
    row["is_correct"] = True
    assert flags(row)["error_then_correct"]
