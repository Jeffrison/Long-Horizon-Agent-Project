"""Export dev-only, outcome-filtered supervision and candidate preferences."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from eval.data_loader import question_key, split_for
from eval.run_eval import read_rows


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def export(run_directory, output_directory):
    source = Path(run_directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = read_rows(source / "episodes.jsonl")
    # Recompute partition from question text: a forged split field is insufficient.
    eligible = [r for r in rows if r["split"] == "dev" and split_for(r["question"]) == "dev"
                and r["question_key"] == question_key(r["question"])]
    if not eligible:
        raise ValueError("No dev trajectories. Test runs must never become training data.")
    sft, preferences, sft_seen, preference_seen = [], [], set(), set()
    branches = defaultdict(list)
    for row in eligible:
        result = row["result"]
        # Require complete protocol-valid trajectories; truncated/API-error episodes are excluded.
        if result["status"] != "success":
            continue
        for event in result.get("events", []):
            if event.get("outcome") not in {"tool_success", "final_answer"} or event.get("finish_reason") == "length":
                continue
            context = event["messages"]
            completion = [{"role": "assistant", "content": event["content"]}]
            metadata = {"question_key": row["question_key"], "episode_key": row["key"],
                        "split": "dev", "source_revision": manifest["git_revision"]}
            if row["is_correct"]:
                key = stable_hash([context, completion])
                if key not in sft_seen:
                    sft_seen.add(key)
                    sft.append({"messages": context + completion, "metadata": metadata})
            branches[(row["question_key"], stable_hash(context))].append(
                (row["is_correct"], context, completion, metadata))
    # Compare only responses conditioned on byte-equivalent full prompts.
    # Final outcome is a noisy label for an earlier decision: candidate pairs require review.
    for branch in branches.values():
        chosen = [x for x in branch if x[0]]
        rejected = [x for x in branch if not x[0]]
        for good in chosen:
            for bad in rejected:
                if good[2] == bad[2] or good[1] != bad[1]:
                    continue
                key = stable_hash([good[1], good[2], bad[2]])
                if key in preference_seen:
                    continue
                preference_seen.add(key)
                preferences.append({"prompt": good[1], "chosen": good[2], "rejected": bad[2],
                    "metadata": {"question_key": good[3]["question_key"], "split": "dev",
                        "chosen_episode": good[3]["episode_key"], "rejected_episode": bad[3]["episode_key"],
                        "label_source": "terminal_answer_correctness", "requires_review": True}})
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("sft.jsonl", sft), ("preference_candidates.jsonl", preferences)):
        (output / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data), encoding="utf-8")
    summary = {"source": str(source), "source_manifest_sha256": hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest(),
               "eligible_dev_episodes": len(eligible), "sft_examples": len(sft),
               "preference_candidates": len(preferences), "question_keys": sorted({r["question_key"] for r in eligible}),
               "limitations": ["Outcome-filtered self-generated SFT is not expert supervision.",
                   "Candidate DPO labels are noisy; review before training.",
                   "No model has been trained by this export."]}
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.run_directory, args.output)
