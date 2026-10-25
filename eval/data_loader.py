"""Fail-fast dataset validation and stable question-level splits."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def question_key(question: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question).casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_for(question: str) -> str:
    # Global by question, independent of file/id/task: duplicate questions cannot cross splits.
    return "dev" if int(question_key(question)[:8], 16) % 5 == 0 else "test"


def load_jsonl_dataset(file_path: str, max_samples: int = None) -> list:
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    dataset, ids, questions = [], set(), set()
    with Path(file_path).open(encoding="utf-8-sig") as stream:
        for line_num, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}") from exc
            question = raw.get("question") or raw.get("problem")
            answer = raw.get("answer")
            item_id = str(raw.get("_id", raw.get("id", f"sample_{line_num}")))
            if not isinstance(question, str) or not question.strip() or answer is None or str(answer).strip() == "":
                raise ValueError(f"Missing question/answer on line {line_num}")
            if not isinstance(answer, (str, int, float)):
                raise ValueError(f"Unsupported answer type on line {line_num}")
            key = question_key(question)
            if item_id in ids or key in questions:
                raise ValueError(f"Duplicate id/question on line {line_num}")
            ids.add(item_id)
            questions.add(key)
            dataset.append({"id": item_id, "question": question, "answer": str(answer),
                            "question_key": key, "split": split_for(question)})
    if not dataset:
        raise ValueError("Empty dataset")
    return dataset if max_samples is None else dataset[:max_samples]
