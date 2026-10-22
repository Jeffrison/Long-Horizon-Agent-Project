"""Stable question identity shared by evaluation and post-training splits."""
import hashlib
import json
from pathlib import Path


def question_key(question):
    canonical = " ".join(str(question).split()).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_for_question(question, seed=42):
    digest = hashlib.sha256(f"{seed}:{question_key(question)}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def dataset_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl_dataset(file_path, max_samples=None, split="all", seed=42):
    if split not in {"all", "train", "validation", "test"}:
        raise ValueError("invalid split")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    samples, seen = [], set()
    with Path(file_path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question = item.get("question", item.get("problem"))
            answer = item.get("answer")
            if not isinstance(question, str) or not question.strip() or answer is None or str(answer) == "":
                raise ValueError(f"Invalid sample at line {line_number}")
            key = question_key(question)
            if key in seen:
                raise ValueError(f"Duplicate question at line {line_number}")
            seen.add(key)
            item_split = split_for_question(question, seed)
            if split != "all" and split != item_split:
                continue
            samples.append({"id": key, "source_id": str(item.get("_id", item.get("id", line_number))),
                            "question": question, "answer": str(answer), "split": item_split})
    return samples[:max_samples] if max_samples else samples
