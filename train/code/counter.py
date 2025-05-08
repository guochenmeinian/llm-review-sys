import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

# 修改为你自己的路径
model_path = "meta-llama/Llama-3.1-8B-Instruct"

ds = load_dataset("guochenmeinian/openreview_dataset", "eval")["train"]

# 加载 tokenizer（LLaMA 专属配置）
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


print(f"📄 加载了 {len(ds)} 条样本")

# 统计 token 长度
token_lengths = []

for example in tqdm(ds, desc="🧮 Tokenizing"):
    prompt = f"{example['instruction'].strip()}\n\n{example['input'].strip()}"
    response = example['output'].strip()
    full_text = f"{prompt}\n\n### Response:\n{response}"
    token_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    token_lengths.append(len(token_ids))

# 输出统计结果
token_lengths = np.array(token_lengths)
print("\n📊 样本 token 长度统计：")
print(f"最大长度: {token_lengths.max()}")
print(f"最小长度: {token_lengths.min()}")
print(f"平均长度: {token_lengths.mean():.2f}")
print(f"中位数长度: {np.median(token_lengths):.2f}")
print(f"90% 样本 ≤ {np.percentile(token_lengths, 90):.0f} tokens")
print(f"95% 样本 ≤ {np.percentile(token_lengths, 95):.0f} tokens")
print(f"99% 样本 ≤ {np.percentile(token_lengths, 99):.0f} tokens")