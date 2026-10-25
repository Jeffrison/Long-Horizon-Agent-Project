"""Verify actual TRL collator masks prompt/observation tokens before trusting SFT."""
import argparse
import json
from pathlib import Path
from post_training.train_sft import prepare_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    import torch
    train, validation = prepare_examples(args.data)
    rows = train + validation
    tokenizer = AutoTokenizer.from_pretrained(args.model,revision=args.revision,trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,
                                                torch_dtype=torch.bfloat16,trust_remote_code=False)
    dataset = Dataset.from_list([{"prompt":r["prompt"],"completion":r["completion"]} for r in rows])
    trainer = SFTTrainer(model=model,processing_class=tokenizer,train_dataset=dataset,
        args=SFTConfig(output_dir=".cache/mask-audit",completion_only_loss=True,packing=False,
                       max_length=4096,report_to="none",bf16=True))
    if len(trainer.train_dataset) != len(rows):
        raise ValueError("Unexpected dataset filtering")
    masked = supervised = 0
    for index,row in enumerate(rows):
        prefix = tokenizer.apply_chat_template(row["prompt"],tokenize=True,add_generation_prompt=True,return_dict=False)
        feature = trainer.train_dataset[index]
        batch = trainer.data_collator([feature])
        labels = batch["labels"][0].tolist()
        if any(label != -100 for label in labels[:len(prefix)]):
            raise ValueError("Prompt or observation tokens are not fully masked")
        targets = sum(label != -100 for label in labels)
        if not targets:
            raise ValueError("No completion targets")
        masked += sum(label == -100 for label in labels)
        supervised += targets
    result = {"examples_checked":len(rows),"prompt_tokens_masked":masked,"completion_tokens_supervised":supervised,
              "all_prompts_and_observations_masked":True,"model":args.model,"revision":args.revision,
              "implementation":"actual TRL-preprocessed dataset and trainer.data_collator"}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

