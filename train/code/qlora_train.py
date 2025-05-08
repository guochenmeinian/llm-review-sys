import os
import argparse
import yaml
import torch
import json
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_config(config_path):
    print(f"📄 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset_from_hf(dataset_repo, file_name):
    print(f"📦 Loading HF dataset: {dataset_repo}/{file_name}")
    dataset = load_dataset(dataset_repo, data_files=file_name, split="train")
    print(f"✅ Loaded {len(dataset)} examples.")
    return dataset

# 构建 prompt
def build_prompt(example):
    return {
        "prompt": f"{example['instruction']}\n\n{example['input']}".strip(),
        "response": example["output"]
    }
    
def preprocess_full_prompt(examples, tokenizer, max_length=18000):
    input_ids, attention_masks, labels = [], [], []

    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i].strip()
        response = examples["response"][i].strip()
        full_text = f"{prompt}\n\n### Response:\n{response}"

        encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        # 使用同样的 truncation 限制计算 prompt 的实际长度，防止越界
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )["input_ids"]

        # prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prompt_len = min(len(prompt_ids), max_length)  # ✅ 限制长度，避免越界
        label_ids = encoding["input_ids"].copy()
        label_ids[:prompt_len] = [-100] * prompt_len

        input_ids.append(encoding["input_ids"])
        attention_masks.append(encoding["attention_mask"])
        labels.append(label_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="qlora_train_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config_file)

    model_path = config["model_name_or_path"]
    output_dir = config["output_dir"]

    print(f"🚀 Loading tokenizer and model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    cutoff_len = tokenizer.model_max_length # 128k for LLaMA3.1 but we only use 18k
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
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
    train_dataset = load_dataset(config["hf_dataset_repo"], "qlora_train")["train"].map(build_prompt)
    val_dataset = load_dataset(config["hf_dataset_repo"], "qlora_validation")["train"].map(build_prompt)

    print("✂️ Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_full_prompt(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda examples: preprocess_full_prompt(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")

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


