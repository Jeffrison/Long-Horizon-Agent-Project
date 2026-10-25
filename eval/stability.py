"""Three-seed analysis on the predeclared 10 questions/task; reuses main seed-11 traces."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from eval.run_eval import read_rows
from eval.report import summarize,paired
from eval.verify_run import verify


def analyze(primary,extra,destination):
    primary,extra,destination=Path(primary),Path(extra),Path(destination)
    if not verify(primary)["complete"] or not verify(extra)["complete"]:
        raise ValueError("Both source experiments must be complete")
    first=json.loads((primary/"manifest.json").read_text(encoding="utf-8"))
    second=json.loads((extra/"manifest.json").read_text(encoding="utf-8"))
    a,b=first["settings"],second["settings"]
    if a["seeds"]!=[11] or b["seeds"]!=[29,47] or a["config"]!=b["config"]:
        raise ValueError("Unexpected seed/configuration plan")
    if a["metric_version"]!=b["metric_version"]:
        raise ValueError("Use the same scorer for both source experiments")
    expected={}
    for task,items in a["tasks"].items():
        expected[task]={x["question_key"] for x in sorted(items,key=lambda x:x["question_key"])[:10]}
        if expected[task]!={x["question_key"] for x in b["tasks"].get(task,[])}:
            raise ValueError("Supplementary sample differs from predeclared hash-first selection")
    selected=[r for r in read_rows(primary/"episodes.jsonl") if r["question_key"] in expected[r["task"]]]
    selected+=read_rows(extra/"episodes.jsonl")
    if len(selected)!=120 or len({r["key"] for r in selected})!=120:
        raise ValueError("Expected 120 unique records, no new sampling")
    result={"questions":20,"episodes":120,"seeds":[11,29,47],"metric_version":a["metric_version"],
            "model_calls_by_analysis":0,"groups":{},"paired":{},
            "source_episode_sha256":{str(p):hashlib.sha256((p/"episodes.jsonl").read_bytes()).hexdigest()
                                     for p in [primary,extra]}}
    lines=["# 预定小子集的三种子稳定性", "",
           "每任务按问题哈希排序的前 10 题。seed=11 直接引用主实验原轨迹；29/47 来自补充实验，不重新生成主实验答案。",
           "", "| Task | Seed | Mode | Correct/N | Accuracy |",
           "|---|---:|---|---:|---:|"]
    for task in sorted(expected):
        task_rows=[r for r in selected if r["task"]==task]
        result["paired"][task]=paired(task_rows)
        for seed in [11,29,47]:
            for mode in ["off","on"]:
                group=summarize([r for r in task_rows if r["seed"]==seed and r["mode"]==mode])
                result["groups"][f"{task}/{seed}/{mode}"]=group
                lines.append(f"| {task} | {seed} | {mode} | {group['correct']}/{group['episodes']} | {group['accuracy']:.2%} |")
    lines+=["", "每个任务只有 10 道题；不能将这些结果声称为完整 215 题的三种子准确率。主实验以全部题目的 seed=11 成对结果为准。",
            "", "配对区间按题目聚类，保留同题三个种子；详细区间、调用与错误统计见 summary.json。"]
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (destination/"report.md").write_text("\n".join(lines),encoding="utf-8")
    data="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in selected).encode("utf-8")
    (destination/"selected-episodes.jsonl.gz").write_bytes(gzip.compress(data,mtime=0))
    print(json.dumps({"questions":20,"episodes":120,"new_model_calls":0},indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("primary")
    parser.add_argument("extra")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    analyze(args.primary,args.extra,args.output)

