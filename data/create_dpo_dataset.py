import json

# create base model dpo dataset
input_path = "rejected_dataset/inference_base_model.jsonl"
output_path = "llama_factory_dataset/dpo_pair_llama3.json"

dpo_data = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        prompt = f"{item['instruction'].strip()}\n\n{item['input'].strip()}"
        chosen = item["output"].strip()
        rejected = item["generated_output"].strip()

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成，共 {len(dpo_data)} 条，已保存到 {output_path}")
