"""Compare base/LoRA completion NLL on question-disjoint DEV validation examples."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

from post_training.train_sft import prepare_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _, validation = prepare_examples(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision,
                                               torch_dtype=torch.bfloat16, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, args.adapter).to("cuda").eval()
    records = []
    started = time.monotonic()
    for row in validation:
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False)
        full = tokenizer.apply_chat_template(row["prompt"]+row["completion"], tokenize=True, return_dict=False)
        if full[:len(prompt)] != prompt:
            raise ValueError("Prompt tokens are not an exact full-sequence prefix")
        inputs = torch.tensor([full], device="cuda")
        labels = inputs.clone()
        labels[:, :len(prompt)] = -100
        supervised = int((labels[:, 1:] != -100).sum())
        if supervised == 0:
            raise ValueError("No completion targets")
        row_result = {"question_key":row["question_key"], "supervised_tokens":supervised,
                      "prompt_tokens_masked":len(prompt)}
        for name, context in (("base", model.disable_adapter), ("lora", nullcontext)):
            with context(), torch.no_grad():
                output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels)
            row_result[name+"_nll"] = float(output.loss)
        records.append(row_result)
    count = sum(r["supervised_tokens"] for r in records)
    summary = {"base_nll":sum(r["base_nll"]*r["supervised_tokens"] for r in records)/count,
               "lora_nll":sum(r["lora_nll"]*r["supervised_tokens"] for r in records)/count,
               "examples":len(records), "questions":len({r["question_key"] for r in records}),
               "completion_tokens":count, "elapsed_seconds":time.monotonic()-started,
               "model":args.model, "revision":args.model_revision,
               "data_sha256":hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
               "scope":"question-disjoint development validation; teacher-forced NLL, NOT answer accuracy",
               "precision":"bfloat16", "records":records}
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="records"},indent=2))


if __name__ == "__main__":
    main()

