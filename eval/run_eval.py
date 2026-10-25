"""Predeclared paired evaluation: one trajectory per question/seed/mode."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.react_loop import AgentConfig, run_agent
from eval.data_loader import load_jsonl_dataset
from eval.metrics import qa_match, qa_f1, math_match

DATASETS = {"QA": "data/validated/hotpotqa200.jsonl", "MATH": "data/validated/math.jsonl"}
_lock = threading.Lock()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, row):
    with _lock:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def read_rows(path):
    if not Path(path).exists():
        return []
    # Never silently discard a damaged last line: repair explicitly, preserving evidence.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def job_key(task, item, seed, mode):
    return f"{task}:{item['id']}:{seed}:{mode}"


def execute_job(task, item, seed, mode, config):
    result = run_agent(item["question"], task_type=task,
                       enable_reflection=(mode == "on"), config=replace(config, seed=seed))
    correct = result["status"] == "success" and (
        qa_match(result["final_answer"], item["answer"]) if task == "QA"
        else math_match(result["final_answer"], item["answer"]))
    return {"key": job_key(task, item, seed, mode), "task": task, "id": item["id"],
            "question_key": item["question_key"], "question": item["question"],
            "split": item["split"], "seed": seed, "mode": mode,
            "ground_truth": item["answer"], "is_correct": bool(correct),
            "answer_f1": qa_f1(result["final_answer"], item["answer"]) if task == "QA" else None,
            "result": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=DATASETS, default=["QA", "MATH"])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or len(args.seeds) != len(set(args.seeds)) or (args.limit is not None and args.limit < 1):
        parser.error("Workers/limit must be positive; seeds must be unique")
    config = AgentConfig(model=args.model, max_calls=args.max_calls, max_tokens=args.max_tokens,
                         history_turns=args.history_turns, temperature=args.temperature, timeout=args.timeout)
    datasets = {}
    for task in dict.fromkeys(args.tasks):
        items = load_jsonl_dataset(DATASETS[task])
        items = [x for x in items if args.split == "all" or x["split"] == args.split]
        # Hash order is stable and does not favor the beginning of a source file.
        datasets[task] = sorted(items, key=lambda x: x["question_key"])[:args.limit]
    directory = Path(args.output)
    manifest_path = directory / "manifest.json"
    settings = {"tasks": datasets, "dataset_sha256": {t: sha(DATASETS[t]) for t in datasets},
                "config": asdict(config), "seeds": args.seeds, "modes": ["off", "on"],
                "split": args.split, "metric_version": "strict-v2",
                "protocol": "one-trajectory-no-answer-selection", "workers": args.workers}
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["settings"] != settings:
            parser.error("Resume settings/data differ from the frozen manifest")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if manifest["git_revision"] != revision:
            parser.error("Source revision changed; resume requires the original implementation")
    else:
        if directory.exists() and any(directory.iterdir()):
            parser.error("Output directory is nonempty; use --resume with identical settings")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
                    "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain",
                                                               "--untracked-files=no"], text=True).strip()),
                    "settings": settings,
                    "limitations": ["Seed is requested; provider determinism is not guaranteed.",
                        "Existing local datasets were used in historical development; test is an operational split, not pristine unseen data.",
                        "QA uses live Wikipedia, not the official HotpotQA distractor protocol."]}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    records_path, started_path = directory / "episodes.jsonl", directory / "started.jsonl"
    rows = read_rows(records_path)
    done = {r["key"] for r in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate evaluation records")
    started_keys = {r["key"] for r in read_rows(started_path)}
    jobs = []
    for task, items in datasets.items():
        for item in items:
            for seed in args.seeds:
                modes = ["off", "on"]
                random.Random(f"{item['id']}:{seed}").shuffle(modes)
                for mode in modes:
                    key = job_key(task, item, seed, mode)
                    if key not in done:
                        jobs.append((task, item, seed, mode))
    random.Random(20261025).shuffle(jobs)
    # An attempt interrupted before checkpoint is counted as a failure, never resampled.
    pending = []
    for job in jobs:
        task, item, seed, mode = job
        key = job_key(*job)
        if key in started_keys:
            append(records_path, {"key": key, "task": task, "id": item["id"], "question_key": item["question_key"],
                "question": item["question"], "split": item["split"], "seed": seed, "mode": mode,
                "ground_truth": item["answer"], "is_correct": False, "answer_f1": 0.0 if task == "QA" else None,
                "result": {"status": "failed", "error": "interrupted", "final_answer": None,
                           "events": [], "trajectory": [], "llm_calls": 0, "latency_seconds": 0}})
        else:
            pending.append(job)
    print(f"Planned {sum(len(x) for x in datasets.values()) * len(args.seeds) * 2} episodes; pending {len(pending)}", flush=True)

    def worker(job):
        key = job_key(*job)
        append(started_path, {"key": key, "started_at": datetime.now(timezone.utc).isoformat()})
        try:
            row = execute_job(*job, config)
        except Exception as exc:
            task, item, seed, mode = job
            row = {"key": key, "task": task, "id": item["id"], "question_key": item["question_key"],
                   "question": item["question"], "split": item["split"], "seed": seed, "mode": mode,
                   "ground_truth": item["answer"], "is_correct": False, "answer_f1": 0.0 if task == "QA" else None,
                   "result": {"status": "failed", "error": "runner_error", "error_type": type(exc).__name__,
                              "final_answer": None, "events": [], "trajectory": [], "llm_calls": 0,
                              "latency_seconds": 0}}
        append(records_path, row)
        return row

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, job) for job in pending]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if index % 10 == 0 or index == len(pending):
                print(f"{index}/{len(pending)} elapsed={time.monotonic()-started:.0f}s last={row['key']} "
                      f"correct={row['is_correct']} status={row['result']['status']}", flush=True)
    from eval.report import write_report
    write_report(directory)
    print(f"Report: {directory / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
