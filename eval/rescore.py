"""Offline rescore of historical QA logs; original files are never modified."""
import argparse
from collections import Counter
import json
from pathlib import Path
from eval.data_loader import dataset_sha256, question_key
from eval.metrics import qa_scores
from eval.provenance import policy_for_file


def rescore(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "MATH" in Path(path).name or data.get("metadata", {}).get("task_type") == "MATH":
        raise ValueError("Historical math logs contain gold-dependent early stopping; excluded.")
    rows = data["logs"]
    scored = [qa_scores(row.get("prediction") if row.get("status") == "success" else "",
                        row["ground_truth"]) for row in rows]
    actual_steps = sum(row.get("steps_taken", row.get("steps", 0)) for row in rows)
    policy = policy_for_file(path, data)
    return {"file": Path(path).name, "sha256": dataset_sha256(path),
            "source_policy": policy,
            "metric_scope": "stored_records_only_not_unbiased_pass_at_1",
            "unbiased_pass_at_1": None, "unbiased_pass_at_k": None,
            "total": len(rows), "unique_questions": len({question_key(r["question"]) for r in rows}),
            "answer_em": sum(s["em"] for s in scored) / len(rows),
            "answer_f1": sum(s["f1"] for s in scored) / len(rows),
            "legacy_custom_accuracy_percent": data.get("metrics", {}).get("accuracy"),
            "actual_mean_recorded_steps": actual_steps / len(rows),
            "assistant_messages": sum(sum(t["role"] == "assistant" for t in r.get("trajectory", [])) for r in rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "warnings": ["Historical ON records were selected after extra attempts; missing attempts are unavailable.",
                         "Rescoring cannot correct outcome-conditioned selection or reconstruct pass@k.",
                         "Historical runs are not controlled by the new call budget.",
                         "Answer metrics only; no supporting-fact/joint metrics.",
                         "No new model inference was performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output", default="outputs/reports/legacy_qa_rescore.json")
    args = parser.parse_args()
    reports = [rescore(path) for path in args.logs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
