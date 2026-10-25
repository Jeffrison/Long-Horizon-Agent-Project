"""Optional local Transformers/PEFT backend; use the separate training environment."""
import time


class LocalChatClient:
    def __init__(self, config, revision="main", adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config = config
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(config.model, revision=revision,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=False)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.revision = revision

    def complete(self, messages, seed):
        self.torch.manual_seed(seed)
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_tensors="pt", return_dict=True)
        encoded = {key:value.to(self.model.device) for key,value in encoded.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, max_new_tokens=self.config.max_tokens,
                do_sample=self.config.temperature > 0,
                **({"temperature":self.config.temperature} if self.config.temperature > 0 else {}),
                pad_token_id=self.tokenizer.eos_token_id)
        prompt_length = encoded["input_ids"].shape[1]
        completion = generated[0, prompt_length:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True).split("Observation:",1)[0].strip()
        return {"content":text, "finish_reason":"length" if len(completion) >= self.config.max_tokens else "stop",
                "usage":{"prompt_tokens":prompt_length,"completion_tokens":len(completion)},
                "model":self.config.model, "requested_seed":seed, "model_revision":self.revision}


