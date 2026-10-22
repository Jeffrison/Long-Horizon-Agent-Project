"""Prospective fixed-budget reflection/restart comparison, with append-only attempts."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from agent.client import APIClient, ScriptedClient
from agent.config import AgentConfig
from agent.prompts import PROMPT_VERSION
from agent.react_loop import run_agent
from eval.data_loader import dataset_sha256, load_jsonl_dataset, question_key
from eval.metrics import qa_scores
from eval.run_eval import code_revision, source_fingerprint
from eval.sampling import pass_at_k, select_without_reference
from tools import build_tools
from tools.types import ToolResult

STRATEGIES = ("react_single", "react_error_feedback", "react_restarts")


def generate_question(query, config, client_factory, tools_factory, *,
                      total_calls=8, restarts=4, sink=None):
    """No reference, scorer, or correctness-dependent stopping exists in this API."""
    if total_calls < 1 or restarts < 1 or total_calls % restarts:
        raise ValueError("Use positive total_calls divisible by restarts")
    output = {}
    for strategy in STRATEGIES:
        count = restarts if strategy == "react_restarts" else 1
        mode = "react_reflect" if strategy == "react_error_feedback" else "react"
        attempts = []
        for index in range(count):
            allocation = total_calls // count
            cfg = replace(config, mode=mode, max_llm_calls=allocation,
                          max_tool_calls=config.max_tool_calls // count,
                          seed=config.seed + index * 1000)
            result = run_agent(query, config=cfg,
                               client=client_factory(strategy, index), tools=tools_factory())
            item = {"strategy": strategy, "attempt_index": index, "seed": cfg.seed,
                    "allocated_llm_calls": allocation,
                    "allocated_tool_calls": cfg.max_tool_calls, "result": result}
            attempts.append(item)
            if sink:
                sink(item)  # Every completed attempt is written, even if it failed.
        if sum(item["result"]["stats"]["llm_calls"] for item in attempts) > total_calls:
            raise RuntimeError("Model budget exceeded")
        output[strategy] = attempts
    return output


def score_question(generated, reference):
    """Scoring happens after generation; oracle coverage never selects the answer."""
    output = {}
    for strategy, attempts in generated.items():
        index = select_without_reference(attempts, "majority" if len(attempts) > 1 else "first")
        selected = attempts[index]["result"] if index is not None else None
        prediction = selected.get("final_answer") if selected and selected["status"] == "success" else ""
        outcomes = [qa_scores(item["result"].get("final_answer")
                              if item["result"]["status"] == "success" else "", reference)
                    for item in attempts]
        correct = sum(int(item["em"]) for item in outcomes)
        known_usage = all(item["result"]["stats"]["prompt_tokens"] is not None for item in attempts)
        output[strategy] = {
            "attempt_count": len(attempts), "selected_index": index,
            "selected_prediction": prediction, "selected_scores": qa_scores(prediction, reference),
            "first_attempt_em": outcomes[0]["em"], "mean_attempt_em": correct / len(attempts),
            "oracle_any_success": bool(correct),
            "pass_at_k": {str(k): pass_at_k(len(attempts), correct, k)
                          for k in range(1, len(attempts) + 1)},
            "actual_llm_calls": sum(item["result"]["stats"]["llm_calls"] for item in attempts),
            "actual_tool_calls": sum(item["result"]["stats"]["tool_calls"] for item in attempts),
            "elapsed_seconds": sum(item["result"]["stats"]["elapsed_seconds"] for item in attempts),
            "prompt_tokens": (sum(item["result"]["stats"]["prompt_tokens"] for item in attempts)
                              if known_usage else None),
            "completion_tokens": (sum(item["result"]["stats"]["completion_tokens"] for item in attempts)
                                  if known_usage else None),
            "oracle_used_for_selection": False,
        }
    return output


def run_comparison(samples, config, client_factory, tools_factory, *, total_calls=8,
                   restarts=4, attempt_sink=None):
    logs, questions = [], []
    for position, item in enumerate(samples):
        cfg = replace(config, seed=config.seed + position * 100000)

        def save(attempt):
            if attempt_sink:
                attempt_sink({"question_id": item["id"], "question": item["question"], **attempt})
        generated = generate_question(item["question"], cfg, client_factory, tools_factory,
                                      total_calls=total_calls, restarts=restarts, sink=save)
        scored = score_question(generated, item["answer"])
        questions.append({"id": item["id"], "question": item["question"],
                          "ground_truth": item["answer"], "strategies": scored})
        for strategy, attempts in generated.items():
            for attempt in attempts:
                result = attempt["result"]
                prediction = result.get("final_answer") if result["status"] == "success" else ""
                scores = qa_scores(prediction, item["answer"])
                logs.append({**result, "id": item["id"], "question": item["question"],
                             "source_id": item.get("source_id"), "split": item["split"],
                             "ground_truth": item["answer"], "prediction": prediction,
                             "scores": scores, "is_correct": bool(scores["em"]), "task_type": "QA",
                             "strategy": strategy, "attempt_index": attempt["attempt_index"]})
    if not questions:
        raise ValueError("No samples")
    metrics = {}
    for strategy in STRATEGIES:
        rows = [q["strategies"][strategy] for q in questions]
        metrics[strategy] = {
            "questions": len(rows),
            "selected_em": sum(r["selected_scores"]["em"] for r in rows) / len(rows),
            "selected_f1": sum(r["selected_scores"]["f1"] for r in rows) / len(rows),
            "first_attempt_em": sum(r["first_attempt_em"] for r in rows) / len(rows),
            "oracle_any_success_rate": sum(r["oracle_any_success"] for r in rows) / len(rows),
            "mean_actual_llm_calls": sum(r["actual_llm_calls"] for r in rows) / len(rows),
            "mean_actual_tool_calls": sum(r["actual_tool_calls"] for r in rows) / len(rows),
            "mean_prompt_tokens": (sum(r["prompt_tokens"] for r in rows) / len(rows)
                                   if all(r["prompt_tokens"] is not None for r in rows) else None),
            "mean_completion_tokens": (sum(r["completion_tokens"] for r in rows) / len(rows)
                                       if all(r["completion_tokens"] is not None for r in rows) else None),
            "mean_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows) / len(rows),
            "oracle_pass_at_k": {key: sum(r["pass_at_k"][key] for r in rows) / len(rows)
                                 for key in rows[0]["pass_at_k"]},
        }
    return {"schema_version": 2, "logs": logs, "questions": questions,
            "metrics_by_strategy": metrics}


def fixture_tools():
    facts = {"testbook": "TestBook was written by Ada Example.",
             "ada example": "Ada Example was born in Exampleland."}

    def search(query):
        text = facts.get(query.strip("\"' ").casefold())
        return ToolResult(bool(text), text or "No matching fixture entity.", None if text else "no_results")
    return {"Search": search}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/hotpotqa200.jsonl")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--total-calls", type=int, default=8)
    parser.add_argument("--tool-budget", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["api", "local", "fixture"], default="fixture")
    parser.add_argument("--model", default="outputs/models/qwen2.5-0.5b-instruct")
    parser.add_argument("--fixture-environment", action="store_true",
                        help="Synthetic two-hop environment; never a real benchmark")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.total_calls < 1 or args.restarts < 1 or args.total_calls % args.restarts:
        parser.error("--total-calls must be positive and divisible by --restarts")
    synthetic = args.backend == "fixture" or args.fixture_environment
    if synthetic:
        query = "Which country is the author of TestBook from?"
        samples = [{"id": question_key(query), "question": query, "answer": "Exampleland",
                    "source_id": "synthetic-demo", "split": "test"}]
    else:
        samples = load_jsonl_dataset(args.data, args.max_samples, args.split, args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output = Path(args.output or f"outputs/fair_eval/{stamp}")
    output.mkdir(parents=True, exist_ok=False)  # A fresh plan never overwrites previous attempts.
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
            "model": args.model, "fixture_only": synthetic, "total_calls_per_question_per_strategy": args.total_calls,
            "total_tool_budget": args.tool_budget, "restarts": args.restarts,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "seed": args.seed, "question_ids": [row["id"] for row in samples],
            "prompt_version": PROMPT_VERSION, "git_revision": code_revision(),
            "source_sha256": source_fingerprint(), "reference_used_for_generation": False,
            "dataset_sha256": None if synthetic else dataset_sha256(args.data)}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    config = AgentConfig(model=args.model, max_tool_calls=args.tool_budget,
                         max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed)
    if args.backend == "fixture":
        def factory(strategy, index):
            return ScriptedClient(["Action: Search\nAction Input: TestBook",
                                   "Action: Search\nAction Input: Ada Example",
                                   "Final Answer: Exampleland"])
    else:
        if args.backend == "local":
            from agent.local_client import LocalHFClient
            client = LocalHFClient(args.model, seed=args.seed)
        else:
            client = APIClient()
        factory = lambda strategy, index: client
    tools_factory = fixture_tools if synthetic else lambda: build_tools("QA")
    with (output / "attempts.jsonl").open("x", encoding="utf-8") as handle:
        def sink(item):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
        report = run_comparison(samples, config, factory, tools_factory,
                                total_calls=args.total_calls, restarts=args.restarts, attempt_sink=sink)
    report["metadata"] = {
        **plan, "task_type": "QA", "completed": True, "generalization_evidence": False,
        "collection_protocol": {"record_selection": "all_attempts", "all_attempts_recorded": True,
                                "reference_used_for_generation": False,
                                "attempt_count_policy": {"react_single": 1, "react_error_feedback": 1,
                                                         "react_restarts": args.restarts}},
        "selection": "first for single trajectories; normalized majority with first-index tie break for restarts",
        "oracle_metric": "diagnostic coverage upper bound only; not a deployable selector",
        "budget_note": "Equal upper call limits, not equal realized input tokens or latency.",
        "reflection_scope": "format/tool-error feedback; no ground-truth correctness feedback",
        "pass_at_k_note": "Assumes complete fixed sampling, comparable independent attempts; not for legacy selected logs.",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "fixture_only": synthetic,
                      "metrics": report["metrics_by_strategy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
