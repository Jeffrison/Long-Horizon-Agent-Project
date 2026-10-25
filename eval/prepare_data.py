"""Preserve raw datasets; assign unique question-hash IDs to validated copies."""
import json
from pathlib import Path
from eval.data_loader import load_jsonl_dataset, question_key


def prepare(output="data/validated"):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    audit = {}
    for task, name in (("QA", "hotpotqa200"), ("MATH", "math")):
        source = Path("data") / (name + ".jsonl")
        rows, seen, source_ids = [], set(), {}
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            query = item.get("question", item.get("problem"))
            key = question_key(query)
            if key in seen:
                raise ValueError(f"Duplicate question in {source}")
            seen.add(key)
            old_id = str(item.get("_id", item.get("id", "")))
            source_ids[old_id] = source_ids.get(old_id, 0) + 1
            rows.append({"id": task.lower() + "-" + key[:16], "question": query,
                         "answer": item["answer"], "source_id": old_id})
        target = directory / (name + ".jsonl")
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to change an existing validated dataset: {target}")
        if not target.exists():
            target.write_bytes(content.encode("utf-8"))
        validated = load_jsonl_dataset(target)
        audit[task] = {"source": source.as_posix(), "validated": target.as_posix(), "questions": len(rows),
                       "repeated_source_ids": sum(n - 1 for n in source_ids.values()),
                       "splits": {s: sum(x["split"] == s for x in validated) for s in ("dev", "test")}}
    (directory / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    prepare()
