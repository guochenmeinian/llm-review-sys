import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from transformers import BitsAndBytesConfig  
from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model, LoraConfig 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===== 初始化 wandb =====
print("📡 初始化 wandb...")
wandb.init(
    project="dpo_llama3_project",
    name="qlora_full_context_llama3_vs_dataset"
)

# ===== 配置 =====
base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
dataset_repo = "guochenmeinian/openreview_dataset"
dpo_split = "dpo_base"
output_dir = "models/dpo_model_base"

# ===== 加载 tokenizer =====
print("🔍 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ===== 配置 QLoRA 量化参数 =====
print("🧬 加载 base model + QLoRA adapter...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ===== 加载 QLoRA 微调权重（adapter） =====
qlora_model_path = "models/full_context_qlora_model"
model = PeftModel.from_pretrained(base_model, qlora_model_path)
print("✅ 成功加载 QLoRA 微调模型")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
print("✅ QLoRA LoRA 配置完成")

# ===== 加载 DPO 数据集 =====
print(f"📦 加载数据集 `{dataset_repo}`, split=`{dpo_split}` ...")
dataset = load_dataset(dataset_repo, dpo_split)["train"]
print(f"✅ 数据集加载完成，共 {len(dataset)} 条样本")

# ===== DPO Trainer 配置 =====
dpo_config = DPOConfig(
    beta=0.1,  # 偏好强度
    max_length=18000,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-6,
    num_train_epochs=3,
    logging_steps=10,
    output_dir=output_dir,
    bf16=True,
    remove_unused_columns=False,
    report_to="wandb",
    run_name="qlora_full_context_llama3_vs_dataset"
)

# ===== 创建 Trainer =====
# 只使用必需的参数
trainer = DPOTrainer(
    model=model,
    eval_dataset=None, # we don't use eval_dataset here
    args=dpo_config,
    train_dataset=dataset,
    ref_model=None, # DPO doesn't require a reference model
    processing_class=tokenizer
)

# ===== 训练 =====
checkpoint_dir = os.path.join(output_dir, "checkpoint-last")
if os.path.isdir(checkpoint_dir):
    print(f"🔄 发现已有 checkpoint，尝试从 {checkpoint_dir} 恢复训练...")
    trainer.train(resume_from_checkpoint=True)
else:
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print("❌ CUDA OOM! 手动清理缓存中...")
        torch.cuda.empty_cache()
        raise e

# ===== 保存结果 =====
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✨ 训练完成，模型已保存至", output_dir)
