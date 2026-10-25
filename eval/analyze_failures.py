"""Post-hoc failure taxonomy; never sends feedback into the Agent."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from eval.metrics import parse_latex_to_float
from eval.run_eval import read_rows
from eval.report import write_report


def primary_reason(row):
    result = row["result"]
    if row["is_correct"]:
        return "correct"
    error = result.get("error")
    if error == "api_error":
        last = result.get("events", [{}])[-1]
        return "api_timeout" if last.get("error_type") == "ReadTimeout" else "api_failure"
    if error:
        return error
    if row["task"] == "MATH":
        return "invalid_numeric_final" if parse_latex_to_float(result.get("final_answer")) is None else "wrong_numeric_final"
    return "qa_partial_match" if row.get("answer_f1",0) > 0 else "qa_no_token_overlap"


def flags(row):
    result = row["result"]
    events = result.get("events",[])
    actions = [(e["tool"]["name"],e["tool"]["input"]) for e in events if "tool" in e]
    return {"had_parse_error":result.get("parse_errors",0)>0,
            "had_tool_error":result.get("tool_errors",0)>0,
            "had_truncation":any(e.get("finish_reason")=="length" for e in events),
            "repeated_tool_input":len(set(actions))<len(actions),
            "error_then_correct":row["is_correct"] and bool(result.get("parse_errors",0)+result.get("tool_errors",0))}


def analyze(directory):
    directory=Path(directory)
    report=write_report(directory)
    if not report["complete"]:
        raise ValueError("Failure analysis requires a complete planned run")
    rows=read_rows(directory/"episodes.jsonl")
    groups=defaultdict(list)
    for row in rows:
        groups[(row["task"],row["mode"])].append(row)
    result={"groups":{}, "paired_examples":{"improved":[],"regressed":[]}}
    lines=["# 错误归因", "", "主分类互斥且覆盖全部轨迹；辅助标记可重叠。错误后答对率是条件描述，不能单独作为反思因果增益。",
           "", "| Task/mode | N | Primary outcomes | Auxiliary flags |","|---|---:|---|---|"]
    for (task,mode),group in sorted(groups.items()):
        primary=dict(Counter(primary_reason(r) for r in group))
        secondary={name:sum(flags(r)[name] for r in group) for name in flags(group[0])}
        result["groups"][task+"/"+mode]={"total":len(group),"primary":primary,"auxiliary":secondary}
        lines.append(f"| {task}/{mode} | {len(group)} | {primary} | {secondary} |")
    paired=defaultdict(dict)
    for row in rows:
        paired[(row["question_key"],row["seed"])][row["mode"]]=row
    for key,group in sorted(paired.items()):
        if set(group)!={"off","on"} or group["off"]["is_correct"]==group["on"]["is_correct"]:
            continue
        category="improved" if group["on"]["is_correct"] else "regressed"
        if len(result["paired_examples"][category])>=3:
            continue
        result["paired_examples"][category].append({"question_key":key[0],"seed":key[1],
            "question":group["off"]["question"],"ground_truth":group["off"]["ground_truth"],
            "off":{k:group["off"]["result"].get(k) for k in ("final_answer","error","llm_calls","parse_errors","tool_errors")},
            "on":{k:group["on"]["result"].get(k) for k in ("final_answer","error","llm_calls","parse_errors","tool_errors")}})
    lines+=["", "成对案例按题目哈希排序选取每类至多 3 条，未按戏剧性筛选。具体文本见 failure-analysis.json。",
            "", "辅助标记不能相加得到失败总数；API 错误和预算耗尽也可能同时伴随先前的解析/工具错误。"]
    (directory/"failure-analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (directory/"failure-analysis.md").write_text("\n".join(lines),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory")
    analyze(parser.parse_args().directory)

