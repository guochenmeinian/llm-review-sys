

# import os
# import argparse
# import yaml
# import torch
# import json
# from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
# from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig

# # 加载 train yaml 的 配置文件
# def load_config(config_path):
#     print(f"📄 Loading config from: {config_path}")
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)

# # 加载数据集
# def load_dataset_from_json(json_path):
#     print(f"📦 Loading dataset from: {json_path}")
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     print(f"✅ Loaded {len(data)} examples.")
#     return Dataset.from_list(data)

# # 构建 prompt
# def build_prompt(example):
#     return {
#         "prompt": f"{example['instruction']}\n\n{example['input']}".strip(),
#         "response": example["output"]
#     }

# # 预处理和填充数据
# def preprocess_function(examples, tokenizer, max_length):
#     # 批量处理多个样本
#     prompts = examples["prompt"]
#     responses = examples["response"]
    
#     # 对提示和响应进行编码
#     prompt_encodings = tokenizer(prompts, truncation=True, max_length=max_length)
#     response_encodings = tokenizer(responses, truncation=True, max_length=max_length)
    
#     # 准备输入和标签
#     input_ids = []
#     attention_mask = []
#     labels = []
    
#     for i in range(len(prompts)):
#         prompt_ids = prompt_encodings["input_ids"][i]
#         prompt_mask = prompt_encodings["attention_mask"][i]
#         response_ids = response_encodings["input_ids"][i]
        
#         # 连接提示和响应
#         combined_ids = prompt_ids + response_ids
#         combined_mask = prompt_mask + [1] * len(response_ids)
        
#         # 创建标签：提示部分为-100，响应部分为实际ID
#         combined_labels = [-100] * len(prompt_ids) + response_ids
        
#         # 截断到最大长度
#         if len(combined_ids) > max_length:
#             combined_ids = combined_ids[:max_length]
#             combined_mask = combined_mask[:max_length]
#             combined_labels = combined_labels[:max_length]
        
#         # 填充到最大长度
#         padding_length = max_length - len(combined_ids)
#         if padding_length > 0:
#             combined_ids = combined_ids + [tokenizer.pad_token_id] * padding_length
#             combined_mask = combined_mask + [0] * padding_length
#             combined_labels = combined_labels + [-100] * padding_length
        
#         input_ids.append(combined_ids)
#         attention_mask.append(combined_mask)
#         labels.append(combined_labels)
    
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_file", type=str, default="model/openreview_qlora_train.yaml")
#     args = parser.parse_args()

#     config = load_config(args.config_file)

#     model_path = config["model_name_or_path"]
#     output_dir = config["output_dir"]
#     train_file = config["train_file"]
#     val_file = config["eval_file"]
#     cutoff_len = 2048  # 使用更合理的上下文长度

#     print(f"🚀 Loading tokenizer and model from: {model_path}")
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.model_max_length = cutoff_len

#     bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
#     model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
#     model.config.rope_scaling = {"type": "dynamic", "factor": cutoff_len / 4096}
#     model = prepare_model_for_kbit_training(model)

#     print("🔧 Applying LoRA configuration...")
#     lora_config = LoraConfig(
#         r=config.get("lora_rank", 8),
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
#     model = get_peft_model(model, lora_config)

#     # ✅ 添加以下两行
#     model.config.use_cache = False  # ⚠️ 关闭缓存，兼容 gradient checkpointing
#     model.gradient_checkpointing_enable()  # ⚠️ 启用 gradient checkpointing

#     print("📊 Processing training and validation datasets...")
#     train_dataset = load_dataset_from_json(train_file).map(build_prompt)
#     val_dataset = load_dataset_from_json(val_file).map(build_prompt)

#     print("✂️ 预处理和填充数据集...")
#     # 使用批处理方式处理数据集，每批处理1000个样本
#     train_dataset = train_dataset.map(
#         lambda examples: preprocess_function(examples, tokenizer, cutoff_len),
#         batched=True,
#         batch_size=1000,
#         remove_columns=train_dataset.column_names
#     )
    
#     val_dataset = val_dataset.map(
#         lambda examples: preprocess_function(examples, tokenizer, cutoff_len),
#         batched=True,
#         batch_size=1000,
#         remove_columns=val_dataset.column_names
#     )
    
#     # 设置数据集格式为torch
#     train_dataset.set_format(type="torch")
#     val_dataset.set_format(type="torch")

#     print("⚙️ Preparing training arguments...")
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
#         per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
#         gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
#         num_train_epochs=config.get("num_train_epochs", 3),
#         eval_strategy="steps",
#         eval_steps=config.get("eval_steps", 100),
#         save_steps=config.get("save_steps", 100),
#         learning_rate=config.get("learning_rate", 2e-5),
#         bf16=True,
#         logging_steps=10,
#         save_total_limit=2,
#         report_to="none",
#         logging_dir=os.path.join(output_dir, "logs"),
#         # 禁用数据加载器的固定长度
#         dataloader_drop_last=False,
#         remove_unused_columns=False
#     )

#     print("🏋️ Starting training...")
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         # 不使用数据整理器
#         data_collator=None
#     )

#     trainer.train()

#     print(f"💾 Saving model and tokenizer to {output_dir}")
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)

#     print("✅ Training complete.")

# if __name__ == "__main__":
#     main()



###### 通过分段处理数据，解决显存不足的问题，需要测试一下

import os
import argparse
import yaml
import torch
import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
# os.environ["HF_HOME"] = "/workspace/hf_home"

# 加载 train yaml 的 配置文件
def load_config(config_path):
    print(f"📄 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset_from_hf(dataset_repo, file_name, split="train"):
    print(f"📦 Loading HF dataset: {dataset_repo}/{file_name}")
    dataset = load_dataset(dataset_repo, data_files=file_name, split=split)
    print(f"✅ Loaded {len(dataset)} examples.")
    return dataset

# 构建 prompt
def build_prompt(example):
    return {
        "prompt": f"{example['instruction']}\n\n{example['input']}".strip(),
        "response": example["output"]
    }

def chunk_and_expand_dataset(dataset, tokenizer, max_length=8192, stride=8192):
    expanded_data = []
    for example in dataset:
        instruction = example["instruction"]
        long_input = example["input"]
        full_output = example["output"]

        prompt = f"{instruction}\n\n{long_input}".strip()
        tokenized = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        for start in range(0, len(tokenized), stride):
            end = start + max_length
            chunk_ids = tokenized[start:end]
            if not chunk_ids:
                continue

            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            expanded_data.append({
                "instruction": instruction,
                "input": chunk_text,
                "output": full_output
            })

    return Dataset.from_list(expanded_data)

def preprocess_full_prompt(examples, tokenizer, max_length=8192):
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

    print("📊 Expanding and processing training and validation datasets...")
    train_dataset = load_dataset(config["hf_dataset_repo"], "qlora_train")["train"].map(build_prompt)
    val_dataset = load_dataset(config["hf_dataset_repo"], "qlora_validation")["train"].map(build_prompt)

    train_dataset = chunk_and_expand_dataset(train_dataset, tokenizer)
    val_dataset = chunk_and_expand_dataset(val_dataset, tokenizer)

    train_dataset = train_dataset.map(build_prompt)
    val_dataset = val_dataset.map(build_prompt)

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
        learning_rate=float(config.get("learning_rate", 2e-5)),
        bf16=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs")
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("🏋️ Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()

    print(f"💾 Saving model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Training complete.")

if __name__ == "__main__":
    main()