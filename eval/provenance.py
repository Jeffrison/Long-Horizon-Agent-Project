"""Content-addressed collection history; selected training data is not a benchmark."""
import json
from pathlib import Path
from eval.data_loader import dataset_sha256


def source_policy(sha256, report=None):
    registry = Path(__file__).resolve().parents[1] / "data" / "source_provenance.json"
    entries = json.loads(registry.read_text(encoding="utf-8"))["sources"]
    if sha256 in entries:
        return dict(entries[sha256])
    protocol = (report or {}).get("metadata", {}).get("collection_protocol", {})
    complete = (protocol.get("record_selection") == "all_attempts"
                and protocol.get("all_attempts_recorded") is True
                and protocol.get("reference_used_for_generation") is False)
    fixture = (report or {}).get("metadata", {}).get("fixture_only", False)
    return {"record_selection": "all_attempts" if complete else "unknown",
            "all_attempts_recorded": True if complete else None,
            "selection_conditioned_training_data": None if not complete else False,
            "eligible_for_reflection_gain_comparison": complete and not fixture,
            "fixture_only": fixture,
            "training_use": "Silver labels still need review; eligibility does not prove fresh test data."}


def policy_for_file(path, report=None):
    return source_policy(dataset_sha256(path), report)
