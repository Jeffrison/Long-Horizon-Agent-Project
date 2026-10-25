import json
from pathlib import Path
import pytest
from eval.data_loader import load_jsonl_dataset, split_for, question_key
from eval.report import paired
from post_training.export import export


def test_zero_answer_and_duplicate_validation(tmp_path):
    path = tmp_path/"data.jsonl"
    path.write_text('{"id":"1","question":"q","answer":0}\n', encoding="utf-8")
    assert load_jsonl_dataset(path)[0]["answer"] == "0"
    path.write_text(path.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_jsonl_dataset(path)


def test_split_ignores_question_case_whitespace():
    assert question_key("  WHAT is X? ") == question_key("what  is x?")
    assert split_for("  WHAT is X? ") == split_for("what  is x?")


def test_clustered_pairs_keep_seed_repetitions_together():
    rows = [{"question_key": "q", "seed": seed, "mode": mode, "is_correct": mode == "on"}
            for seed in [1, 2, 3] for mode in ["off", "on"]]
    result = paired(rows)
    assert result["unique_questions"] == 1
    assert result["complete_pairs"] == 3
    assert result["delta_accuracy"] == 1


def test_training_export_refuses_test_data(tmp_path):
    run = tmp_path/"run"
    run.mkdir()
    (run/"manifest.json").write_text('{"git_revision":"abc"}')
    question = next(f"q{i}" for i in range(100) if split_for(f"q{i}") == "test")
    row = {"split":"test", "question":question, "question_key":question_key(question)}
    (run/"episodes.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="No dev"):
        export(run, tmp_path/"out")


def test_preference_prompt_alignment_and_sft_filtering(tmp_path):
    run = tmp_path/"run"
    run.mkdir()
    (run/"manifest.json").write_text('{"git_revision":"abc"}')
    question = next(f"q{i}" for i in range(100) if split_for(f"q{i}") == "dev")
    context = [{"role":"user","content":question}]
    rows = []
    for index, (correct, prompt) in enumerate([(True, context), (False, context),
                                              (False, [{"role":"user","content":"DIFFERENT"}])]):
        rows.append({"split":"dev", "question":question, "question_key":question_key(question),
            "key":str(index), "is_correct":correct,
            "result":{"status":"success","events":[{"outcome":"final_answer","messages":prompt,
                "content":f"Final Answer: {index}","finish_reason":"stop"}]}})
    (run/"episodes.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
    result = export(run,tmp_path/"out")
    assert result["sft_examples"] == 1 and result["preference_candidates"] == 1
