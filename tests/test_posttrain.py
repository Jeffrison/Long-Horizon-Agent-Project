import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.data_loader import split_for_question
from posttrain.build_data import export_data
from posttrain.train import load_training_data, main as train_main


def row(question, prediction):
    return {"question": question, "ground_truth": "Paris", "prediction": prediction,
            "status": "success", "trajectory": []}


class DataExportTests(unittest.TestCase):
    def test_same_question_cannot_cross_splits_and_pairs_share_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "Eval_QA_fixture.json"
            questions = [f"capital question {i}" for i in range(80)]
            source = {"logs": [row(q, pred) for q in questions for pred in ("Paris", "London")]}
            path.write_text(json.dumps(source))
            manifest = export_data([path], root / "output")
            owners = {}
            for family in ("sft", "dpo"):
                for split in ("train", "validation", "test"):
                    for line in (root / "output" / f"{family}_{split}.jsonl").read_text().splitlines():
                        item = json.loads(line)
                        previous = owners.setdefault(item["question_id"], split)
                        self.assertEqual(previous, split)
                        if family == "dpo":
                            self.assertNotEqual(item["chosen"], item["rejected"])
            with self.assertRaises(ValueError):
                load_training_data(root / "output", "sft")
            data, _ = load_training_data(root / "output", "sft", accept_silver=True)
            self.assertTrue(data["train"])
            self.assertFalse(manifest["training_completed"])
            captured = io.StringIO()
            with patch("sys.argv", ["train", "--method", "sft", "--data-dir", str(root / "output"),
                                     "--dry-run", "--accept-silver-labels"]), contextlib.redirect_stdout(captured):
                train_main()
            self.assertFalse(json.loads(captured.getvalue())["training_completed"])

    def test_math_logs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Eval_MATH_fixture.json"
            path.write_text(json.dumps({"logs": []}))
            with self.assertRaises(ValueError):
                export_data([path], Path(directory) / "output")

    def test_legacy_context_is_not_fabricated_for_agent_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Eval_QA_fixture.json"
            path.write_text(json.dumps({"logs": [row("q", "Paris")]}))
            with self.assertRaises(ValueError):
                export_data([path], Path(directory) / "output", kind="decision")

    def test_preferences_require_identical_decision_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "v2.json"
            records = []
            for evidence, answer in (("Evidence A", "Paris"), ("Evidence B", "London")):
                item = row("q", answer)
                item["events"] = [{"kind": "llm", "messages": [{"role": "user", "content": evidence}],
                                   "text": "Final Answer: " + answer,
                                   "parsed": {"type": "finish", "content": answer}}]
                records.append(item)
            path.write_text(json.dumps({"schema_version": 2, "metadata": {"task_type": "QA"}, "logs": records}))
            manifest = export_data([path], root / "output", kind="decision")
            self.assertEqual(sum(v for k, v in manifest["counts"].items() if k.startswith("dpo")), 0)


if __name__ == "__main__":
    unittest.main()
