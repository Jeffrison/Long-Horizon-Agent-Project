"""Optional local Transformers/LoRA backend for base-vs-adapter QA evaluation."""
class LocalHFClient:
    def __init__(self, model_name, adapter=None, seed=42):
        import os
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        if not torch.cuda.is_available():
            raise RuntimeError("Local evaluation expects CUDA-enabled PyTorch")
        set_seed(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to("cuda")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.model.config.use_cache = True
        self.model_name = model_name
        self.calls = 0

    def complete(self, messages, config):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        length = inputs["input_ids"].shape[-1]
        if length + config.max_new_tokens > getattr(self.model.config, "max_position_embeddings", 32768):
            raise ValueError("Context too long; reduce history/snippet lengths")
        self.torch.manual_seed(config.seed + self.calls)
        self.calls += 1
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=config.max_new_tokens,
                                            do_sample=config.temperature > 0,
                                            **({"temperature": config.temperature} if config.temperature > 0 else {}),
                                            pad_token_id=self.tokenizer.pad_token_id)
        tokens = generated[0, length:]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        text = text.split("Observation:", 1)[0].strip()
        return {"text": text, "usage": {"prompt_tokens": length, "completion_tokens": len(tokens)},
                "response_model": self.model_name}
