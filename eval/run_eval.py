"""CLI evaluation with direct/react/react_reflect baselines and no gold leakage."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.client import APIClient
from agent.config import AgentConfig
from agent.prompts import PROMPT_VERSION
from agent.react_loop import run_agent
from eval.data_loader import dataset_sha256, load_jsonl_dataset
from eval.metrics import math_match, qa_scores


def summarize(rows):
    count = len(rows)
    if not count:
        raise ValueError("No evaluation rows")
    completed = sum(row["status"] == "success" for row in rows)
    calls = sum(row["stats"]["llm_calls"] for row in rows)
    tools = sum(row["stats"]["tool_calls"] for row in rows)
    successful_tools = sum(row["stats"]["tool_successes"] for row in rows)
    latencies = sorted(row["stats"]["elapsed_seconds"] for row in rows)
    error_counts = {}
    for row in rows:
        for key, value in row["stats"]["errors"].items():
            error_counts[key] = error_counts.get(key, 0) + value
    usages = all(row["stats"]["prompt_tokens"] is not None for row in rows)
    return {"total": count, "answer_rate": completed / count,
            "em": sum(row["scores"]["em"] for row in rows) / count,
            "f1": (sum(row["scores"]["f1"] for row in rows) / count
                   if all(row["scores"]["f1"] is not None for row in rows) else None),
            "avg_llm_calls": calls / count, "avg_tool_calls": tools / count,
            "tool_success_rate": successful_tools / tools if tools else None,
            "format_error_rate_per_call": error_counts.get("format_error", 0) / calls if calls else 0,
            "avg_seconds": sum(latencies) / count,
            "p95_seconds": latencies[min(count - 1, int(0.95 * count))],
            "prompt_tokens": sum(row["stats"]["prompt_tokens"] for row in rows) if usages else None,
            "completion_tokens": sum(row["stats"]["completion_tokens"] for row in rows) if usages else None,
            "error_counts": error_counts}


def code_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("agent", "tools", "eval"):
        for path in sorted((root / folder).glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate(dataset_path, task_type, config, *, max_samples=None, split="test",
             client=None, tools=None, allow_python=False, delay=0.0, backend_details=None):
    samples = load_jsonl_dataset(dataset_path, max_samples, split, config.seed)
    if not samples:
        raise ValueError("Selected split contains no samples")
    client = client or APIClient()
    rows = []
    for item in samples:
        # Inference sees the question, configuration and tools; never item["answer"].
        result = run_agent(item["question"], task_type, config=config, client=client,
                           tools=tools, allow_python=allow_python)
        prediction = result["final_answer"] if result["status"] == "success" else ""
        scores = qa_scores(prediction, item["answer"]) if task_type == "QA" else {
            "em": float(math_match(prediction, item["answer"])), "f1": None}
        rows.append({**result, **{k: item[k] for k in ("id", "source_id", "question", "split")},
                     "ground_truth": item["answer"], "prediction": prediction, "scores": scores,
                     "is_correct": bool(scores["em"]), "task_type": task_type})
        if delay > 0:
            time.sleep(delay)
    return {"schema_version": 2, "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(), "task_type": task_type,
                "dataset_name": Path(dataset_path).name, "dataset_sha256": dataset_sha256(dataset_path),
                "split": split, "config": config.to_dict(), "prompt_version": PROMPT_VERSION,
                "git_revision": code_revision(), "source_sha256": source_fingerprint(),
                "backend": backend_details or {"kind": type(client).__name__},
                "collection_protocol": {"record_selection": "all_attempts", "all_attempts_recorded": True,
                                        "reference_used_for_generation": False, "attempts_per_question": 1},
                "metric": "hotpot_answer_em_f1" if task_type == "QA" else "scalar_math_accuracy",
                "notes": ["Existing 200 QA questions were previously evaluated; splits are retrospective.",
                          "Wikipedia is live and may change. Preserve evidence in events.",
                          "Equal upper call budgets do not imply equal realized token cost."]},
            "metrics": summarize(rows), "logs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/hotpotqa200.jsonl")
    parser.add_argument("--task", choices=["QA", "MATH"], default="QA")
    parser.add_argument("--mode", choices=["direct", "react", "react_reflect", "all"], default="all")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-llm-calls", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument("--adapter", help="Local PEFT adapter directory; local backend only")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--allow-python", action="store_true")
    parser.add_argument("--output", default="outputs/eval")
    args = parser.parse_args()
    if args.adapter and args.backend != "local":
        parser.error("--adapter requires --backend local")
    if args.task == "MATH" and args.mode != "direct" and not args.allow_python:
        parser.error("MATH tool use requires --allow-python (local execution is not a security sandbox)")
    modes = ["direct", "react", "react_reflect"] if args.mode == "all" else [args.mode]
    config = AgentConfig(model=args.model, max_llm_calls=args.max_llm_calls,
                         max_tool_calls=args.max_tool_calls, max_new_tokens=args.max_new_tokens,
                         history_turns=args.history_turns, seed=args.seed, temperature=args.temperature)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    if args.backend == "local":
        from agent.local_client import LocalHFClient
        client = LocalHFClient(args.model, args.adapter, args.seed)
    else:
        client = APIClient()
    for mode in modes:
        report = evaluate(args.data, args.task, replace(config, mode=mode),
                          max_samples=args.max_samples, split=args.split, client=client,
                          allow_python=args.allow_python, delay=args.delay,
                          backend_details={"kind": args.backend, "model": args.model,
                                           "adapter": args.adapter})
        path = output / f"{args.task}_{args.split}_{mode}_{stamp}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"mode": mode, "metrics": report["metrics"], "path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
