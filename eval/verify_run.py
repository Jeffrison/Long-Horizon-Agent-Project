"""Verify saved episodes against frozen labels, budgets, counters and scorer."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from eval.metrics import qa_match,qa_f1,math_match
from eval.run_eval import read_rows,job_key


def verify(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    settings=manifest["settings"]
    planned={job_key(task,item,seed,mode):(task,item,seed,mode)
             for task,items in settings["tasks"].items() for item in items
             for seed in settings["seeds"] for mode in settings["modes"]}
    rows=read_rows(directory/"episodes.jsonl")
    seen=set()
    for row in rows:
        key=row["key"]
        if key not in planned or key in seen:
            raise ValueError("Duplicate or unplanned episode")
        seen.add(key)
        task,item,seed,mode=planned[key]
        fields={"task":task,"id":item["id"],"seed":seed,"mode":mode,"question":item["question"],
                "question_key":item["question_key"],"split":item["split"],"ground_truth":item["answer"]}
        if any(row.get(name)!=value for name,value in fields.items()):
            raise ValueError("Episode labels/identity differ from frozen manifest")
        result=row["result"]
        score=(qa_match if task=="QA" else math_match)(result.get("final_answer"),item["answer"])
        if row["is_correct"] != bool(score and result["status"]=="success"):
            raise ValueError("Stored correctness differs from independent scoring")
        if task=="QA" and abs(row["answer_f1"]-qa_f1(result.get("final_answer"),item["answer"]))>1e-12:
            raise ValueError("Stored QA F1 differs from independent scoring")
        events=result.get("events",[])
        if result["llm_calls"]!=len(events) or result["llm_calls"]>settings["config"]["max_calls"]:
            raise ValueError("Inference call accounting/budget mismatch")
        if result.get("error") in {"interrupted","runner_error"}:
            continue
        expected_config={**settings["config"],"seed":seed}
        if result.get("config")!=expected_config or result.get("enable_reflection")!=(mode=="on"):
            raise ValueError("Trajectory configuration differs from frozen plan")
        if result["initial_messages"][1]["content"]!=item["question"]:
            raise ValueError("Agent query differs from frozen question")
        counters={"tool_calls":sum("tool" in e for e in events),
                  "tool_errors":sum(e.get("tool",{}).get("is_error",False) for e in events),
                  "parse_errors":sum(e.get("outcome")=="parse_error" for e in events)}
        counters.update({name:sum(e.get("usage",{}).get(name,0) for e in events)
                         for name in ["prompt_tokens","completion_tokens"]})
        if any(result.get(name)!=value for name,value in counters.items()):
            raise ValueError("Trajectory counters disagree with raw events")
    summary={"verified_episodes":len(rows),"planned_episodes":len(planned),"complete":seen==set(planned),
             "frozen_labels_match":True,"scores_recomputed":True,"budgets_and_counters_match":True}
    (directory/"verification.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory")
    verify(parser.parse_args().directory)

