"""Extract only unlabeled candidate paragraphs from the public HotpotQA snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

from eval.data_loader import load_jsonl_dataset, question_key

SOURCE_URL = "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet"


def prepare(source, output="data/contexts/hotpotqa_distractor.jsonl"):
    # Optional data-preparation dependency; runtime search is pure Python.
    import pyarrow.parquet as pq
    table = pq.ParquetFile(source).read()
    originals = {x["question_key"]: x for x in load_jsonl_dataset("data/validated/hotpotqa200.jsonl")}
    extracted = {}
    for row in table.to_pylist():
        key = question_key(row["question"])
        if key not in originals:
            continue
        if str(row["answer"]) != originals[key]["answer"]:
            raise ValueError("Snapshot answer differs from the existing dataset")
        context = row["context"]
        docs = [{"title": title, "text": " ".join(sentences)}
                for title, sentences in zip(context["title"], context["sentences"])]
        # No answer, supporting_facts, label, or source difficulty enters the tool file.
        extracted[key] = {"question_key": key, "documents": docs}
    if set(extracted) != set(originals):
        raise ValueError("Snapshot does not cover every original QA question")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(extracted[key], ensure_ascii=False)+"\n" for key in sorted(extracted))
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise ValueError("Refusing to alter an existing context snapshot")
    if not target.exists():
        target.write_bytes(content.encode("utf-8"))
    metadata = {"source_url": SOURCE_URL, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "context_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "questions": len(extracted), "documents_per_question": sorted({len(x["documents"]) for x in extracted.values()}),
                "visible_fields": ["question_key", "documents.title", "documents.text"],
                "license": "CC BY-SA 4.0", "attribution": "HotpotQA (Yang et al., EMNLP 2018), Wikipedia contributors",
                "reference": "https://hotpotqa.github.io/",
                "protocol": "question-scoped candidate-paragraph retrieval; no supporting-fact labels exposed"}
    metadata_path=target.with_suffix(".metadata.json")
    metadata_text=json.dumps(metadata, indent=2)
    if not metadata_path.exists() or metadata_path.read_text(encoding="utf-8")!=metadata_text:
        metadata_path.write_bytes(metadata_text.encode("utf-8"))
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", default="data/contexts/hotpotqa_distractor.jsonl")
    args = parser.parse_args()
    prepare(args.source, args.output)
