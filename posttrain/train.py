"""Small-model LoRA SFT/DPO entry. --dry-run needs only the Python standard library."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform

from eval.data_loader import dataset_sha256


def load_training_data(directory, method, accept_silver=False):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    output = {}
    for split in ("train", "validation"):
        filename = f"{method}_{split}.jsonl"
        path = root / filename
        if dataset_sha256(path) != manifest["file_sha256"][filename]:
            raise ValueError("Dataset changed after export. Re-export it (or create a reviewed manifest).")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if row["split"] != split or manifest["question_splits"].get(row["question_id"]) != split:
                raise ValueError("Question split mismatch")
            if row.get("review_status") != "approved" and not accept_silver:
                raise ValueError("Rule labels are unreviewed; review them or explicitly use --accept-silver-labels")
        output[split] = rows
    train_ids = {r["question_id"] for r in output["train"]}
    validation_ids = {r["question_id"] for r in output["validation"]}
    if train_ids & validation_ids:
        raise ValueError("Question leakage across splits")
    if not output["train"]:
        raise ValueError("No training examples. Never fabricate preference pairs to fill the dataset.")
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["sft", "dpo"], required=True)
    parser.add_argument("--data-dir", default="outputs/posttrain_answer")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="outputs/sft_adapter")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accept-silver-labels", action="store_true")
    parser.add_argument("--export-merged", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_length < 32 or args.max_steps < 1:
        parser.error("Use a positive training budget and max-length >= 32")
    rows, manifest = load_training_data(args.data_dir, args.method, args.accept_silver_labels)
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate")}
    plan = {"method": args.method, "model": args.model, "kind": manifest["kind"],
            "train_examples": len(rows["train"]), "validation_examples": len(rows["validation"]),
            "max_steps": args.max_steps, "max_length": args.max_length,
            "lora_rank": 8, "micro_batch": 1, "gradient_accumulation": 8,
            "completion_only_sft": args.method == "sft", "dependencies": dependencies,
            "dry_run": args.dry_run, "training_completed": False}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not all(dependencies.values()):
        raise SystemExit("Missing optional dependencies. Install posttrain/requirements.txt in a training venv.")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("This recipe expects your NVIDIA GPU. CUDA-enabled PyTorch is required.")
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    set_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fields = ["prompt", "completion"] if args.method == "sft" else ["prompt", "chosen", "rejected"]

    def fits(row):
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        for key in fields[1:]:
            full = tokenizer.apply_chat_template(row["prompt"] + row[key], tokenize=True)
            if len(full) > args.max_length or len(full) <= len(prompt):
                return False
        return True

    datasets, dropped = {}, {}
    for split in ("train", "validation"):
        selected = [r for r in rows[split] if fits(r)]
        dropped[split] = len(rows[split]) - len(selected)
        datasets[split] = Dataset.from_list([{k: r[k] for k in fields} for r in selected]) if selected else None
    if datasets["train"] is None:
        raise SystemExit("All sequences exceed max-length; increase it or shorten the prompts.")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    common = dict(output_dir=args.output, per_device_train_batch_size=1,
                  per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                  gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant": False},
                  bf16=bf16, fp16=not bf16, max_steps=args.max_steps,
                  max_length=args.max_length, seed=args.seed, data_seed=args.seed,
                  logging_steps=1, save_strategy="no", report_to="none",
                  dataloader_num_workers=0, optim="adamw_torch")
    if args.method == "sft":
        config = SFTConfig(**common, learning_rate=1e-4,
                           completion_only_loss=True, packing=False)
        trainer_type = SFTTrainer
    else:
        config = DPOConfig(**common, learning_rate=1e-5, beta=0.1,
                           precompute_ref_log_probs=True)
        trainer_type = DPOTrainer
    trainer = trainer_type(model=model, args=config, train_dataset=datasets["train"],
                           eval_dataset=datasets["validation"], processing_class=tokenizer,
                           peft_config=lora)
    result = trainer.train()
    validation = trainer.evaluate() if datasets["validation"] is not None else None
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    if args.export_merged:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(Path(args.output) / "merged")
        tokenizer.save_pretrained(Path(args.output) / "merged")
    audit = {**plan, "dry_run": False, "training_completed": True,
             "train_metrics": result.metrics, "validation_metrics": validation,
             "dropped_overlength": dropped,
             "torch_cuda_max_memory_bytes": torch.cuda.max_memory_allocated(),
             "gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
             "versions": {name: importlib.metadata.version(name) for name in dependencies},
             "data_hashes": manifest["file_sha256"],
             "note": "Training loss is not QA accuracy; evaluate the saved adapter separately."}
    (Path(args.output) / "training_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
