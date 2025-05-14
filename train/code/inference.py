import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel

# set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.environ["HF_HOME"] = "/workspace/hf_home"

# model selection
MODEL_CHOICES = {
    "llama3___1": {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "output_file": "eval/results_eval_llama3___1.jsonl"
    },
    "full_context_qlora": {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "qlora_path": "models/full_context_qlora",
        "output_file": "eval/results_eval_full_context_qlora.jsonl"
    },
    "sliding_window_qlora": {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "qlora_path": "models/sliding_window_qlora",
        "output_file": "eval/results_eval_sliding_window_qlora.jsonl"
    },
    "full_context_qlora_dpo": {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "qlora_path": "models/full_context_dpo",
        "output_file": "eval/results_eval_full_context_qlora_dpo.jsonl"
    },
    "sliding_window_qlora_dpo": {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "qlora_path": "models/sliding_window_dpo",
        "output_file": "eval/results_eval_sliding_window_qlora_dpo.jsonl"
    },
    "full_context_qlora_as_rejected": {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "qlora_path": "models/full_context_dpo_qlora_as_rejected",
        "output_file": "eval/results_eval_full_context_dpo_qlora_as_rejected.jsonl"
    }
}

# select model (manually modify this key)
selected_model = "llama3___1"

config = MODEL_CHOICES[selected_model]
base_model_name = config["base"]
qlora_model_path = config["qlora_path"] if "qlora_path" in config else None
output_path = config["output_file"]

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


if "qlora_path" in config:
    print(f"🔹 Loading QLoRA model from {qlora_model_path} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    model = PeftModel.from_pretrained(base_model, qlora_model_path).eval()
else:
    print(f"🔹 Loading base model only from {base_model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    ).eval()

# load dataset
dataset = load_dataset("guochenmeinian/openreview_dataset", "eval")["train"]
print(f"📄 loaded {len(dataset)} samples")


# inference function
def run_inference(instruction, input_text, max_input_tokens=18000, max_output_tokens=1500):
    prompt = f"{instruction.strip()}\n\n{input_text.strip()}\n\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()



# get processed keys
processed_keys = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                key = record["instruction"].strip() + "||" + record["input"].strip()
                processed_keys.add(key)
            except Exception as e:
                print(f"⚠️ failed to parse line, skipped: {e}")

print(f"🔄 already processed {len(processed_keys)} samples, will skip these samples")

# start inference
with open(output_path, "a", encoding="utf-8") as f_out:
    for i, example in enumerate(tqdm(dataset, desc="🚀 inferring", total=len(dataset))):
        try:
            key = example["instruction"].strip() + "||" + example["input"].strip()
        except Exception as e:
            print(f"⚠️ sample {i+1} format error: {e}")
            continue

        if key in processed_keys:
            continue  # skip processed samples

        try:
            gen_output = run_inference(example["instruction"], example["input"])
        except Exception as e:
            print(f"❌ sample {i+1} inference failed: {e}")
            gen_output = "ERROR"

        result = {
            "instruction": example["instruction"],
            "input": example["input"],
            "output": example["output"],
            "generated_output": gen_output
        }

        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        f_out.flush()

print(f"\n✅ inference completed, results saved to: {output_path}")