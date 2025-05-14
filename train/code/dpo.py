import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from transformers import BitsAndBytesConfig  
from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model, LoraConfig 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===== initialize wandb =====
print("📡 initialize wandb...")
wandb.init(
    project="dpo_llama3_qlora_project",
    name="qlora_full_context model inference as rejected"
)

# ===== config =====
base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
dataset_repo = "guochenmeinian/openreview_dataset"
dpo_split = "dpo_qlora"
output_dir = "models/full_context_qlora_as_rejected"

# ===== load tokenizer =====
print("🔍 load tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ===== configure QLoRA quantization parameters =====
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

# ===== load QLoRA micro-tuning weights (adapter) =====
qlora_model_path = "models/full_context_qlora"
model = PeftModel.from_pretrained(base_model, qlora_model_path, is_trainable=True)
model.train()
print("✅ successfully loaded QLoRA micro-tuning model")

# ===== load DPO dataset =====
print(f"📦 load dataset `{dataset_repo}`, split=`{dpo_split}` ...")
dataset = load_dataset(dataset_repo, dpo_split)["train"]
print(f"✅ dataset loaded, total {len(dataset)} samples")

# ===== DPO Trainer config =====
dpo_config = DPOConfig(
    beta=0.1,  
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

# ===== create trainer =====
# only use necessary parameters
trainer = DPOTrainer(
    model=model,
    eval_dataset=None, # we don't use eval_dataset here
    args=dpo_config,
    train_dataset=dataset,
    ref_model=None, # DPO doesn't require a reference model
    processing_class=tokenizer
)

# ===== training =====
checkpoint_dir = os.path.join(output_dir, "checkpoint-last")
if os.path.isdir(checkpoint_dir):
    print(f"🔄 found existing checkpoint, trying to resume training from {checkpoint_dir}...")
    trainer.train(resume_from_checkpoint=True)
else:
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print("❌ CUDA OOM! Manually clearing cache...")
        torch.cuda.empty_cache()
        raise e

# ===== save result =====
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✨ training complete, model saved to", output_dir)
