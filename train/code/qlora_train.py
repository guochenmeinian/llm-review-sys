import os
import argparse
import yaml
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig

# 加载 train yaml 的 配置文件
def load_config(config_path):
    print(f"📄 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 加载数据集
def load_dataset_from_json(json_path):
    print(f"📦 Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} examples.")
    return Dataset.from_list(data)

# 构建 prompt
def build_prompt(example):
    return {
        "prompt": f"{example['instruction']}\n\n{example['input']}".strip(),
        "response": example["output"]
    }

def tokenize_example(example, tokenizer, cutoff_len):
    prompt = tokenizer(
        example["prompt"], max_length=cutoff_len, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        label = tokenizer(
            example["response"], max_length=cutoff_len, truncation=True, padding="max_length"
        )
    prompt["labels"] = label["input_ids"]
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="model/openreview_qlora_train.yaml")
    args = parser.parse_args()

    config = load_config(args.config_file)

    model_path = config["model_name_or_path"]
    output_dir = config["output_dir"]
    train_file = config["train_file"]
    val_file = config["eval_file"]

    print(f"🚀 Loading tokenizer and model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    cutoff_len = tokenizer.model_max_length
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    print("🔧 Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=config.get("lora_rank", 8),
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # ✅ 添加以下两行
    model.config.use_cache = False  # ⚠️ 关闭缓存，兼容 gradient checkpointing
    model.gradient_checkpointing_enable()  # ⚠️ 启用 gradient checkpointing

    print("📊 Processing training and validation datasets...")
    train_dataset = load_dataset_from_json(train_file).map(build_prompt)
    val_dataset = load_dataset_from_json(val_file).map(build_prompt)

    print("✂️ Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda ex: tokenize_example(ex, tokenizer, cutoff_len), batched=True)
    val_dataset = val_dataset.map(lambda ex: tokenize_example(ex, tokenizer, cutoff_len), batched=True)

    print("⚙️ Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        num_train_epochs=config.get("num_train_epochs", 3),
        eval_strategy="steps",
        eval_steps=config.get("eval_steps", 100),
        save_steps=config.get("save_steps", 100),
        learning_rate=config.get("learning_rate", 2e-5),
        bf16=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs")
    )

    print("🏋️ Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    print(f"💾 Saving model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Training complete.")

if __name__ == "__main__":
    main()
