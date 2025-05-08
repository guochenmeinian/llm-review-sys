import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset

# 模型路径
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # 或者你的微调模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)


data = load_dataset("guochenmeinian/openreview_dataset", "dpo")["train"]
print(f"📄 数据集中共有 {len(data)} 篇论文样本")


def run_inference(instruction, input_text, max_input_tokens=128000, max_output_tokens=512):
    # print(tokenizer.__class__)
    # print(tokenizer.pad_token, tokenizer.pad_token_id)
    # print(tokenizer.eos_token, tokenizer.eos_token_id)
    # print(tokenizer.vocab_size)  # 对于 LLaMA3，应该是 128256
    # print(tokenizer.special_tokens_map)


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
    
    generated_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text.strip()

# 输出文件路径（.jsonl，每行一个完整 JSON 对象）
output_path = "inference_results_from_dpo_dataset.jsonl"

with open(output_path, "a", encoding="utf-8") as f_out:
    for i, example in enumerate(data):
        print(f"\n===== 推理第 {i+1} 条样本 =====")
        try:
            gen_output = run_inference(example["instruction"], example["input"])
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            gen_output = "ERROR"

        result = {
            "instruction": example["instruction"],
            "input": example["input"],
            "output": example["output"],
            "generated_output": gen_output
        }

        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        f_out.flush()  # 🛡️ 强制写入磁盘，防止奔溃丢失

print(f"\n✅ 推理完成，结果已写入 → {output_path}")