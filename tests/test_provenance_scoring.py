from eval.provenance import hash_sources
from eval.scoring import score_answer


def test_source_fingerprint_changes_on_code_but_not_checkout_newlines(tmp_path):
    file=tmp_path/"agent.py"
    file.write_bytes(b"x = 1\n")
    first=hash_sources([file],tmp_path)
    file.write_bytes(b"x = 1\r\n")
    assert hash_sources([file],tmp_path)==first
    file.write_bytes(b"x = 2\n")
    assert hash_sources([file],tmp_path)!=first


def test_metric_versions_explicitly_preserve_legacy_precision_behavior():
    from eval.metrics import math_match
    assert not math_match("1000001","1000000")
    assert score_answer("1000001","1000000","MATH","strict-v2")
    assert not score_answer("1000001","1000000","MATH","strict-v3")
    assert score_answer("The United States.","United States","QA","strict-v3")
