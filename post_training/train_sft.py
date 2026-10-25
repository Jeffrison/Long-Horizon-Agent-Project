"""Optional LoRA SFT entrypoint; dry-run validation needs no GPU dependencies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from eval.data_loader import split_for, question_key
from eval.run_eval import read_rows


def prepare_examples(path):
    examples = read_rows(path)
    if not examples:
        raise ValueError("Empty SFT file")
    unique = set()
    clean = []
    for row in examples:
        metadata = row["metadata"]
        messages = row["messages"]
        questions = [m["content"] for m in messages if m["role"] == "user"]
        query = questions[0] if questions else ""
        key = question_key(query)
        if metadata.get("split") != "dev" or metadata["question_key"] != key or split_for(query) != "dev":
            raise ValueError("Training input contains a non-dev or mismatched question")
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("SFT example must end with an assistant completion")
        identity = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if identity in unique:
            continue
        unique.add(identity)
        clean.append({"prompt": messages[:-1], "completion": messages[-1:], "question_key": key})
    keys = sorted({x["question_key"] for x in clean})
    if len(keys) < 2:
        raise ValueError("At least two distinct dev questions are needed for train/validation separation")
    validation_keys = set(keys[:max(1, len(keys)//5)])
    train = [x for x in clean if x["question_key"] not in validation_keys]
    validation = [x for x in clean if x["question_key"] in validation_keys]
    return train, validation


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/lora-sft")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    train, validation = prepare_examples(args.data)
    summary = {"train_examples": len(train), "validation_examples": len(validation),
               "train_questions": len({x["question_key"] for x in train}),
               "validation_questions": len({x["question_key"] for x in validation}),
               "input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
               "dry_run": args.dry_run}
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return summary
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this 7B LoRA recipe; use --dry-run for data validation")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision,
                                                torch_dtype=torch.bfloat16, trust_remote_code=False)
    # Reject long examples before training; silent right truncation may discard the completion.
    for row in train + validation:
        tokens = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
        if len(tokens) > args.max_length:
            raise ValueError("Example exceeds max-length; curate or increase the limit before training")
    config = SFTConfig(output_dir=str(output), max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, max_length=args.max_length, completion_only_loss=True,
        packing=False, bf16=True, gradient_checkpointing=True,
        logging_steps=5, eval_strategy="steps", eval_steps=25, save_steps=25,
        save_total_limit=2, report_to="none", seed=args.seed, data_seed=args.seed,
        push_to_hub=False)
    def dataset(rows):
        return Dataset.from_list([{"prompt": x["prompt"], "completion": x["completion"]} for x in rows])
    trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer,
        train_dataset=dataset(train), eval_dataset=dataset(validation),
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules="all-linear", task_type="CAUSAL_LM"))
    output.mkdir(parents=True, exist_ok=True)
    summary.update(model=args.model, requested_model_revision=args.model_revision,
                   resolved_model_revision=getattr(model.config, "_commit_hash", None),
                   arguments=vars(args), training_status="started")
    (output/"data_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trainer.train()
    trainer.save_model(str(output/"adapter"))
    tokenizer.save_pretrained(str(output/"adapter"))
    summary["training_status"] = "completed"
    (output/"data_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
