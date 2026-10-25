import json
import pytest
from pathlib import Path
from eval import run_eval
from eval.metrics import qa_f1


def test_resume_never_resamples_completed_or_interrupted_jobs(tmp_path, monkeypatch):
    data = tmp_path/"qa.jsonl"
    data.write_text('{"id":"q","question":"Question?","answer":"yes"}\n', encoding="utf-8")
    output = tmp_path/"run"
    monkeypatch.setitem(run_eval.DATASETS, "QA", str(data))
    calls = []
    def execute(task, item, seed, mode, config, metric_version="strict-v3"):
        calls.append(mode)
        return {"key":run_eval.job_key(task,item,seed,mode), "task":task, "id":item["id"],
                "question_key":item["question_key"], "question":item["question"],
                "split":item["split"], "seed":seed, "mode":mode, "ground_truth":item["answer"],
                "is_correct":True, "answer_f1":1.0,
                "result":{"status":"success", "error":None,"final_answer":"yes","events":[],
                          "trajectory":[],"llm_calls":1,"latency_seconds":0}}
    monkeypatch.setattr(run_eval,"execute_job",execute)
    args = ["--tasks","QA","--split","all","--seeds","1","--workers","1","--output",str(output)]
    run_eval.main(args)
    assert len(calls) == 2
    run_eval.main(args+["--resume"])
    assert len(calls) == 2
    rows = run_eval.read_rows(output/"episodes.jsonl")
    # Simulate a trajectory that started but whose completion was lost in a crash.
    (output/"episodes.jsonl").write_text(json.dumps(rows[0])+"\n",encoding="utf-8")
    run_eval.main(args+["--resume"])
    assert len(calls) == 2
    restored = run_eval.read_rows(output/"episodes.jsonl")
    assert len(restored) == 2
    assert sum(r["result"].get("error") == "interrupted" for r in restored) == 1
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    assert summary["complete"]
    monkeypatch.setattr(run_eval,"source_fingerprint",lambda:"changed-source")
    with pytest.raises(SystemExit):
        run_eval.main(args+["--resume"])
    assert len(calls)==2
