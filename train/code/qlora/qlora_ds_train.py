import os
import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import torch.distributed.tensor.parallel._utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.distributed.tensor.parallel._utils._DTensor_DispatchMode__enabled = False

def load_config(config_path):
    print(f"\U0001F4C4 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_prompt(example):
    return {
        "prompt": f"{example['instruction']}\n\n{example['input']}".strip(),
        "response": example["output"]
    }

def preprocess_full_prompt(examples, tokenizer, max_length=4096):
    input_ids, attention_masks, labels = [], [], []
    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i].strip()
        response = examples["response"][i].strip()
        full_text = f"{prompt}\n\n### Response:\n{response}"

        encoding = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")
        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]

        prompt_len = min(len(prompt_ids), max_length)
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

def build_lora_config(config):
    return LoraConfig(
        r=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM"
    )

def build_training_args(config, output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        num_train_epochs=config.get("num_train_epochs", 3),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 100),
        save_steps=config.get("save_steps", 100),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        save_total_limit=config.get("save_total_limit", 2),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=config.get("report_to", "none"),
        run_name=config.get("wandb_run_name", None),
        deepspeed=config.get("deepspeed", "ds_config.json")
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="qlora_train_config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)  # ⬅️ Deepspeed 需要这个参数
    parser.add_argument("--deepspeed", type=str, default=None)  # ⬅️ 必须声明这个，否则会报错
    args = parser.parse_args()

    config = load_config(args.config_file)
    model_path = config["model_name_or_path"]
    output_dir = config["output_dir"]
    cutoff_len = config.get("cutoff_len", 4096)

    print(f"\U0001F680 Loading tokenizer and model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # device_map=device_map, # device_map={"": 0}
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model = prepare_model_for_kbit_training(model)

    print("\U0001F527 Applying LoRA configuration...")
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("\U0001F4CA Processing datasets...")
    train_dataset = load_dataset(config["hf_dataset_repo"], "qlora_train")["train"].map(build_prompt)
    val_dataset = load_dataset(config["hf_dataset_repo"], "qlora_validation")["train"].map(build_prompt)

    train_dataset = train_dataset.map(
        lambda examples: preprocess_full_prompt(examples, tokenizer, max_length=cutoff_len),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda examples: preprocess_full_prompt(examples, tokenizer, max_length=cutoff_len),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")

    training_args = build_training_args(config, output_dir)

    print("\U0001F3CB️ Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    print(f"\U0001F4BE Saving model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Training complete.")

if __name__ == "__main__":
    main()
