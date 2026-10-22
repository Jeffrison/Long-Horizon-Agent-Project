"""Collect small reproducible audit summaries without model calls."""
import json
from pathlib import Path
import re
import subprocess
import sys
from scripts.download_model import file_hash
from eval.provenance import source_policy


def main():
    output = Path("reports")
    output.mkdir(exist_ok=True)
    rescore = Path("outputs/reports/legacy_qa_rescore.json")
    manifest = Path("outputs/posttrain_answer_v2/manifest.json")
    if not manifest.exists():
        manifest = Path("outputs/posttrain_answer/manifest.json")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           capture_output=True, text=True, timeout=60)
    fixture = subprocess.run([sys.executable, "-m", "examples.offline_demo"],
                             capture_output=True, text=True, timeout=30)
    count = re.search(r"Ran (\d+) tests", tests.stderr)
    if tests.returncode or fixture.returncode:
        raise RuntimeError("Verification failed; no passing audit report was generated")
    report = {
        "date": "2026-10-22",
        "verification": {"offline_regression_tests": int(count.group(1)) if count else None,
                         "offline_fixture_passed": fixture.returncode == 0},
        "legacy_qa_rescore": json.loads(rescore.read_text(encoding="utf-8")),
        "answer_data": {k: v for k, v in json.loads(manifest.read_text(encoding="utf-8")).items()
                        if k in {"kind", "seed", "counts", "sources", "file_sha256", "limitations",
                                 "purpose", "contains_selection_conditioned_or_unknown_sources",
                                 "eligible_for_reflection_gain_comparison"}},
        "training": {},
        "evaluation": [],
        "latest_evaluation": {},
        "comparison_integrity": None,
        "artifact_sha256": {},
        "historical_collection_disclosure": {
            "source": "project author's 2026-10-22 clarification",
            "on_collection": "extra sampling with correctness-conditioned retention",
            "missing_attempts_recoverable": False,
            "historical_scores_support_reflection_gain": False,
            "selected_demonstrations_can_be_training_data": True,
            "completed_sft_dpo_are_workflow_experiments_not_effectiveness_evidence": True},
        "limitations": [
            "Historical QA rescoring is not an effect of the new implementation.",
            "Dataset splits are retrospective, not a fresh unseen benchmark.",
            "Answer-only training does not directly train tool selection.",
            "Preference labels are unreviewed silver labels; DPO validation split is empty.",
            "Small training/eval runs validate the pipeline, not general model improvement.",
        ],
    }
    for name in ("sft_adapter", "dpo_adapter"):
        path = Path("outputs") / name / "training_report.json"
        report["training"][name] = (json.loads(path.read_text(encoding="utf-8"))
                                     if path.exists() else {"training_completed": False})
        matched_manifest = None
        selection_conditioned = None
        for candidate in (Path("outputs/posttrain_answer/manifest.json"),
                          Path("outputs/posttrain_answer_v2/manifest.json")):
            if candidate.exists():
                data_manifest = json.loads(candidate.read_text(encoding="utf-8"))
                if data_manifest["file_sha256"] == report["training"][name].get("data_hashes"):
                    matched_manifest = str(candidate)
                    policies = [source_policy(s["sha256"]).get("selection_conditioned_training_data")
                                for s in data_manifest["sources"]]
                    selection_conditioned = True if any(p is True for p in policies) else (
                        False if all(p is False for p in policies) else None)
                    break
        report["training"][name]["retrospective_source_audit"] = {
            "contains_selection_conditioned_demonstrations": selection_conditioned,
            "matched_data_manifest": matched_manifest,
            "audit_does_not_modify_training_report": True,
            "interpretation": "Valid small-scale training execution; not unbiased reflection evidence"}
    for name in ("outputs/models/qwen2.5-0.5b-instruct/model.safetensors",
                 "outputs/sft_adapter/adapter_model.safetensors",
                 "outputs/sft_adapter/merged/model.safetensors",
                 "outputs/dpo_adapter/adapter_model.safetensors"):
        path = Path(name)
        if path.exists():
            report["artifact_sha256"][name] = file_hash(path)
    for folder in ("base_eval", "sft_eval", "dpo_eval"):
        for path in sorted((Path("outputs") / folder).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            report["evaluation"].append({"file": str(path), "metadata": data["metadata"],
                                         "metrics": data["metrics"]})
            report["latest_evaluation"][folder] = report["evaluation"][-1]
    latest = list(report["latest_evaluation"].values())
    if len(latest) == 3:
        raw = [json.loads(Path(item["file"]).read_text(encoding="utf-8")) for item in latest]
        ids = [[row["id"] for row in item["logs"]] for item in raw]
        keys = ("mode", "max_llm_calls", "max_tool_calls", "max_new_tokens", "temperature", "seed")
        settings = [{key: item["metadata"]["config"][key] for key in keys} for item in raw]
        report["comparison_integrity"] = {
            "same_question_ids": all(value == ids[0] for value in ids),
            "same_prompt_version": len({item["metadata"]["prompt_version"] for item in raw}) == 1,
            "same_budget_and_decoding": all(value == settings[0] for value in settings),
        }
    report["fair_protocol_runs"] = []
    for path in sorted(Path("outputs/fair_eval").glob("*/report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        report["fair_protocol_runs"].append({
            "file": str(path), "metadata": data["metadata"],
            "metrics": data["metrics_by_strategy"],
            "use": "protocol smoke only" if data["metadata"].get("fixture_only") else "controlled exploratory comparison"})
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"training": {k: v["training_completed"] for k, v in report["training"].items()},
                      "evaluation_runs": len(report["evaluation"])}))


if __name__ == "__main__":
    main()

