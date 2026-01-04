from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class Generator:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            max_memory={
                0: "6GiB",      # GPU (adjust if you know your VRAM)
                "cpu": "16GiB", # CPU RAM
            },
        )

    def generate(self, prompt: str, max_new_tokens=300):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        print("▶ Generating answer...", flush=True)

        output = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

        print("✔ Answer generated", flush=True)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    