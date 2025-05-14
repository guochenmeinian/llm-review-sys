import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

# modify to your own path
model_path = "meta-llama/Llama-3.1-8B-Instruct"

ds = load_dataset("guochenmeinian/openreview_dataset", "eval")["train"]

# load tokenizer (LLaMA specific config)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


print(f"📄 loaded {len(ds)} samples")

# count token length
token_lengths = []

for example in tqdm(ds, desc="🧮 Tokenizing"):
    prompt = f"{example['instruction'].strip()}\n\n{example['input'].strip()}"
    response = example['output'].strip()
    full_text = f"{prompt}\n\n### Response:\n{response}"
    token_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    token_lengths.append(len(token_ids))

# print statistics
token_lengths = np.array(token_lengths)
print("\n📊 token length statistics:")
print(f"max length: {token_lengths.max()}")
print(f"min length: {token_lengths.min()}")
print(f"mean length: {token_lengths.mean():.2f}")
print(f"median length: {np.median(token_lengths):.2f}")
print(f"90% samples ≤ {np.percentile(token_lengths, 90):.0f} tokens")
print(f"95% samples ≤ {np.percentile(token_lengths, 95):.0f} tokens")
print(f"99% samples ≤ {np.percentile(token_lengths, 99):.0f} tokens")