import json
from eval.data_loader import split_for, question_key
from post_training.train_sft import prepare_examples


def test_sft_validation_has_no_question_overlap(tmp_path):
    questions = [f"training-{i}" for i in range(100) if split_for(f"training-{i}") == "dev"][:6]
    rows = [{"messages":[{"role":"user","content":q},{"role":"assistant","content":"Final Answer: 1"}],
             "metadata":{"split":"dev","question_key":question_key(q)}} for q in questions for _ in range(2)]
    path = tmp_path/"sft.jsonl"
    path.write_text("".join(json.dumps(x)+"\n" for x in rows), encoding="utf-8")
    train, validation = prepare_examples(path)
    assert {x["question_key"] for x in train}.isdisjoint({x["question_key"] for x in validation})
    assert len(train)+len(validation) == 6
