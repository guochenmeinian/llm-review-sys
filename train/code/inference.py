import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from peft import PeftModel

model_name = "meta-llama/Llama-3.1-8B-Instruct" 
qlora_model_path = "../../models/ctx18000_model/outputs-1700/checkpoint-1700"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print("🔹 Loading QLoRA fine-tuned model...")
base_for_qlora = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()
qlora_model = PeftModel.from_pretrained(base_for_qlora, qlora_model_path).eval()


data = load_dataset("guochenmeinian/openreview_dataset", "dpo_base")["train"]
print(f"📄 数据集中共有 {len(data)} 篇论文样本")


def run_inference(instruction, input_text, max_input_tokens=18000, max_output_tokens=1500):
    prompt = f"{instruction.strip()}\n\n{input_text.strip()}\n\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(qlora_model.device)

    with torch.no_grad():
        output_ids = qlora_model.generate(
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
output_path = "model_dpo_results.jsonl"

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
