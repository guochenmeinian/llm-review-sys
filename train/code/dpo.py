import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
import wandb

# ===== 配置 =====
qlora_model_path = "models/full_context_qlora_model"  # 替换为你的 QLoRA 模型路径
dataset_repo = "guochenmeinian/openreview_dataset"
dpo_split = "dpo_base"
output_dir = "models/dpo_model_base"

# ===== 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(qlora_model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ===== 加载 QLoRA 模型 =====
base_model = AutoModelForCausalLM.from_pretrained(
    qlora_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 如果你用了 PEFT 微调（LoRA），可解注释下面行
# base_model = PeftModel.from_pretrained(base_model, qlora_model_path)

# ===== 加载 DPO 数据集 =====
dataset = load_dataset(dataset_repo, dpo_split)["train"]

# ===== DPO Trainer 配置 =====
dpo_config = DPOConfig(
    beta=0.1,  # 偏好强度
    max_length=18000,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-6,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    output_dir=output_dir,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    report_to="wandb",
    run_name="qlora_full_context_llama3_vs_dataset"
)

# ===== 创建 Trainer =====
trainer = DPOTrainer(
    model=base_model,
    ref_model=None,  # 如果你想设为 base 模型的 frozen copy，可指定
    args=dpo_config,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None  # 你可以给个 eval_set，也可以省略
)

# ===== 训练 =====
checkpoint_dir = os.path.join(output_dir, "checkpoint-last")
if os.path.isdir(checkpoint_dir):
    print(f"🔄 发现已有 checkpoint，尝试从 {checkpoint_dir} 恢复训练...")
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# ===== 保存结果 =====
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
