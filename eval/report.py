"""Auditable descriptive metrics and question-clustered paired uncertainty."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from eval.run_eval import read_rows


def wilson(correct, total):
    if not total:
        return [0, 1]
    z = 1.959963984540054
    p = correct / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [max(0, center-radius), min(1, center+radius)]


def summarize(rows):
    total = len(rows)
    correct = sum(r["is_correct"] for r in rows)
    results = [r["result"] for r in rows]
    def avg(name):
        return sum(r.get(name, 0) for r in results) / total if total else 0
    return {"episodes": total, "correct": correct, "accuracy": correct/total if total else 0,
            "unique_questions": len({r["question_key"] for r in rows}),
            "answer_f1": statistics.mean(r["answer_f1"] for r in rows) if rows and rows[0]["task"] == "QA" else None,
            "completion_rate": sum(r["status"] == "success" for r in results)/total if total else 0,
            "avg_llm_calls": avg("llm_calls"), "avg_tool_calls": avg("tool_calls"),
            "avg_latency_seconds": avg("latency_seconds"),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in results),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in results),
            "parse_errors": sum(r.get("parse_errors", 0) for r in results),
            "tool_errors": sum(r.get("tool_errors", 0) for r in results),
            "reflection_events": sum(r.get("reflection_events", 0) for r in results),
            "terminal_failures": dict(Counter(r["error"] for r in results if r.get("error"))),
            "tool_failure_types": dict(Counter(e["tool"]["error_type"] for r in results
                for e in r.get("events", []) if e.get("tool", {}).get("is_error"))),
            "truncated_completions": sum(e.get("finish_reason") == "length" for r in results for e in r.get("events", []))}


def paired(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["question_key"], row["seed"])][row["mode"]] = row
    complete = [g for g in groups.values() if set(g) == {"off", "on"}]
    clusters = defaultdict(list)
    improved = regressed = 0
    for (question, seed), group in groups.items():
        if set(group) != {"off", "on"}:
            continue
        delta = int(group["on"]["is_correct"]) - int(group["off"]["is_correct"])
        clusters[question].append(delta)
        improved += delta == 1
        regressed += delta == -1
    differences = [statistics.mean(values) for values in clusters.values()]
    rng = random.Random(20261025)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(3000)) if differences else [0]*3000
    return {"complete_pairs": len(complete), "unpaired": len(groups) - len(complete),
            "unique_questions": len(clusters), "improved": improved, "regressed": regressed,
            "delta_accuracy": statistics.mean(differences) if differences else 0,
            "cluster_bootstrap_95": [samples[74], samples[2924]],
            "uncertainty_unit": "question; retain all seeds within a resampled question"}


def write_report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = read_rows(directory / "episodes.jsonl")
    settings = manifest["settings"]
    expected = {f"{task}:{item['id']}:{seed}:{mode}" for task, items in settings["tasks"].items()
                for item in items for seed in settings["seeds"] for mode in settings["modes"]}
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError("Duplicate or unplanned records")
    # Recompute labels from saved predictions rather than trusting is_correct fields.
    from eval.metrics import qa_match, math_match
    for row in rows:
        score = (qa_match if row["task"] == "QA" else math_match)(row["result"].get("final_answer"), row["ground_truth"])
        if bool(score and row["result"]["status"] == "success") != row["is_correct"]:
            raise ValueError("Stored score differs from metric recomputation")
    report = {"planned_episodes": len(expected), "completed_episodes": len(rows),
              "complete": set(keys) == expected, "groups": {}, "paired": {}}
    lines = ["# 真实复测报告", "", f"- Source revision: `{manifest['git_revision']}`",
             f"- Protocol: `{settings['protocol']}`; split: `{settings['split']}`",
             f"- Progress: {len(rows)}/{len(expected)} episodes; complete: {report['complete']}",
             f"- Seeds requested: {settings['seeds']}; model: `{settings['config']['model']}`",
             f"- QA retrieval: `{settings['config'].get('retrieval', 'live')}`; context hash: `{settings.get('context_sha256')}`",
             "", "每个题目、种子、模式只运行一条轨迹，所有失败计入分母。success 仅表示产出了协议合法的最终答案。",
             "", "| Task | Mode | Correct / N | Accuracy | Answer F1 | Avg calls | Avg seconds |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task in sorted(settings["tasks"]):
        task_rows = [r for r in rows if r["task"] == task]
        for mode in settings["modes"]:
            group = summarize([r for r in task_rows if r["mode"] == mode])
            report["groups"][task + "/" + mode] = group
            f1 = f"{group['answer_f1']:.3f}" if group["answer_f1"] is not None else "—"
            lines.append(f"| {task} | {mode} | {group['correct']}/{group['episodes']} | "
                         f"{group['accuracy']:.2%} | {f1} | {group['avg_llm_calls']:.2f} | {group['avg_latency_seconds']:.2f} |")
        comparison = paired(task_rows)
        report["paired"][task] = comparison
        for seed in settings["seeds"]:
            for mode in settings["modes"]:
                subset = [r for r in task_rows if r["seed"] == seed and r["mode"] == mode]
                group = summarize(subset)
                group["wilson_95"] = wilson(group["correct"], group["episodes"])
                report["groups"][f"{task}/{mode}/seed-{seed}"] = group
    lines += ["", "## 成对差异与误差分析", ""]
    for task, comparison in report["paired"].items():
        lo, hi = comparison["cluster_bootstrap_95"]
        lines.append(f"- {task}: ON − OFF = {comparison['delta_accuracy']*100:.2f} pp; "
                     f"question-cluster bootstrap 95% [{lo*100:.2f}, {hi*100:.2f}] pp; "
                     f"improved={comparison['improved']}, regressed={comparison['regressed']}.")
    for name, group in report["groups"].items():
        if "/seed-" in name:
            continue
        lines.append(f"- {name}: parse errors={group['parse_errors']}, tool errors={group['tool_errors']}, "
                     f"terminal failures={group['terminal_failures']}, tool failure types={group['tool_failure_types']}.")
    lines += ["", "## 解释边界", "",
        "- 重复种子不能当作独立的新题；主差异区间按题目聚类重采样。每种子 Wilson 区间在 summary.json 中。",
        "- API seed 为请求参数，不能保证服务商确定性；未做择优、重试到正确或基于答案的提前停止。",
        "- 数据来自原项目，曾用于历史调试；test 是本次冻结的操作划分，不宣称完全未见过。",
        "- QA 使用 manifest 中声明的检索后端；snapshot 为问题对应的固定候选段落上的 BM25，live 为实时 Wikipedia。只评答案 EM/F1，不评 supporting facts / joint score，不能对标官方完整榜单。",
        "- 旧日志存在人工挑选，且协议、判分和实现均有变化，本报告不能用于推断相对旧版本的因果增益。",
        "- API/工具故障保留并计为失败；费用以供应商账单为准，日志只记录 token 用量。",
        "- 区间跨零时，不应声称反思有显著增益；即使不跨零，也仅适用于这里的数据和配置。",
        ""]
    (directory / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    write_report(parser.parse_args().directory)
