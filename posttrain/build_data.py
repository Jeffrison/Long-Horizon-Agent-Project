"""Export rule-labeled answer or exact-context Agent decision data.
Legacy QA logs can provide answer-level examples only. Math/oracle logs are refused.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from agent.parser import check_gibberish
from eval.data_loader import dataset_sha256, question_key, split_for_question
from eval.metrics import qa_match

ANSWER_SYSTEM = "Answer the factual question. Return exactly one line: Final Answer: short answer."


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def usable(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        return False
    try:
        check_gibberish(text)
    except ValueError:
        return False
    return True


def export_data(log_paths, output_dir, *, kind="answer", seed=42):
    if kind not in {"answer", "decision"}:
        raise ValueError("kind must be answer or decision")
    candidates = defaultdict(list)
    sft_candidates = []
    sources = []
    questions = {}
    for name in log_paths:
        path = Path(name)
        report = json.loads(path.read_text(encoding="utf-8"))
        metadata = report.get("metadata", {})
        task = metadata.get("task_type")
        if task is None:
            task = "MATH" if "MATH" in path.name else "QA" if "_QA_" in path.name else None
        if task != "QA":
            raise ValueError("Only verified QA logs are accepted; math/oracle trajectories are excluded")
        if kind == "decision" and report.get("schema_version") != 2:
            raise ValueError("Decision export needs schema-v2 logs with exact request messages")
        sources.append({"file": path.name, "sha256": dataset_sha256(path),
                        "legacy": report.get("schema_version") != 2})
        for row in report["logs"]:
            question, answer = row["question"], str(row["ground_truth"])
            key = question_key(question)
            if key in questions and questions[key]["answer"] != answer:
                raise ValueError("Conflicting references for the same question")
            questions[key] = {"question": question, "answer": answer}
            base = {"question_id": key, "split": split_for_question(question, seed),
                    "source_log": path.name, "review_status": "pending",
                    "label_source": "exact_match_rule", "dataset_kind": kind}
            prediction = row.get("prediction")
            correct = row.get("status") == "success" and qa_match(prediction, answer)
            if kind == "answer":
                if row.get("status") != "success" or not usable(prediction):
                    continue
                prompt = [{"role": "system", "content": ANSWER_SYSTEM},
                          {"role": "user", "content": question}]
                response = [{"role": "assistant", "content": "Final Answer: " + prediction.strip()}]
                candidates[(key, fingerprint(prompt))].append(
                    {**base, "prompt": prompt, "response": response, "correct": correct})
            else:
                events = row.get("events", [])
                for index, event in enumerate(events):
                    parsed = event.get("parsed", {})
                    if event.get("kind") != "llm" or event.get("error") or not parsed:
                        continue
                    prompt = event.get("messages")
                    if not isinstance(prompt, list) or not prompt:
                        continue
                    response = [{"role": "assistant", "content": event["text"]}]
                    example = {**base, "prompt": prompt, "completion": response}
                    valid_action = (parsed.get("type") == "action" and index + 1 < len(events)
                                    and events[index + 1].get("kind") == "tool"
                                    and events[index + 1].get("ok") is True)
                    if correct and (valid_action or parsed.get("type") == "finish"):
                        sft_candidates.append(example)
                    # Preference pairs must share the exact inference prefix.
                    # Only final decisions are outcome-labeled; no incorrect action credit.
                    if parsed.get("type") == "finish" and usable(parsed.get("content")):
                        candidates[(key, fingerprint(prompt))].append(
                            {**base, "prompt": prompt, "response": response,
                             "correct": qa_match(parsed["content"], answer)})
    sft, dpo = [], []
    for group in candidates.values():
        positives = [x for x in group if x["correct"]]
        negatives = [x for x in group if not x["correct"]]
        if not positives:
            continue
        chosen = min(positives, key=lambda x: len(x["response"][0]["content"]))
        common = {k: chosen[k] for k in ("question_id", "split", "review_status", "label_source", "dataset_kind")}
        if kind == "answer":
            sft.append({**common, "source_log": chosen["source_log"],
                        "prompt": chosen["prompt"], "completion": chosen["response"]})
        if negatives:
            rejected = min(negatives, key=lambda x: len(x["response"][0]["content"]))
            dpo.append({**common, "source_logs": [chosen["source_log"], rejected["source_log"]],
                        "prompt": chosen["prompt"], "chosen": chosen["response"],
                        "rejected": rejected["response"]})
    if kind == "decision":
        seen = set()
        for item in sft_candidates:
            key = fingerprint([item["prompt"], item["completion"]])
            if key not in seen:
                seen.add(key)
                sft.append(item)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts, files = {}, {}
    for family, examples in (("sft", sft), ("dpo", dpo)):
        for split in ("train", "validation", "test"):
            subset = sorted((x for x in examples if x["split"] == split),
                            key=lambda x: (x["question_id"], fingerprint(x)))
            path = output / f"{family}_{split}.jsonl"
            path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in subset), encoding="utf-8")
            counts[path.name] = len(subset)
            files[path.name] = dataset_sha256(path)
    manifest = {"schema_version": 1, "kind": kind, "seed": seed, "sources": sources,
                "unique_questions": len(questions), "counts": counts, "file_sha256": files,
                "question_splits": {key: split_for_question(item["question"], seed)
                                    for key, item in sorted(questions.items())},
                "split_policy": "question-hash-80-10-10; all attempts for one question stay together",
                "label_policy": "silver EM labels; pending human review; EM may reject correct aliases",
                "training_completed": False,
                "limitations": ["Small retrospective dataset; old runs have seen the full question pool.",
                                "Answer data trains answer behavior, not tool selection or Agentic RL.",
                                "Decision data uses exact saved prompts and assistant-only completions.",
                                "Preference labels need review for aliases and ambiguous answers."]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/posttrain_answer")
    parser.add_argument("--kind", choices=["answer", "decision"], default="answer")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = export_data(args.logs, args.output, kind=args.kind, seed=args.seed)
    print(json.dumps({"kind": manifest["kind"], "counts": manifest["counts"],
                      "training_completed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
