from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
from datasets import load_dataset
from prefix_tuning import setup_prefix_tuning
import json

def setup_model(use_prefix=False):
    """设置模型，可选择是否使用Prefix Tuning"""
    if use_prefix:
        return setup_prefix_tuning()
    else:
        # 原有的QLora设置
        return setup_qlora_model()

from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

def setup_qlora_model(model_name="meta-llama/Llama-3-8b-instruct"):
    """设置QLoRA模型"""
    # 加载基础模型并进行量化
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config={
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    )
    
    # 准备模型进行4bit训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        inference_mode=False
    )
    
    return get_peft_model(model, peft_config)

def prepare_dataset(data_path, parsed_texts_dir=None):
    """准备训练数据集，结合论文全文与评审数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建训练样本
    processed_data = []
    for item in data:
        paper_id = item.get('id', '')
        title = item.get('title', '')
        conference = item.get('conference', '')
        
        # 尝试加载对应的论文全文
        paper_content = ""
        if parsed_texts_dir and paper_id:
            # 查找对应会议目录下的论文文件
            conf_dir = os.path.join(parsed_texts_dir, conference)
            if os.path.exists(conf_dir):
                paper_file = os.path.join(conf_dir, f"{paper_id}.mmd")
                if os.path.exists(paper_file):
                    with open(paper_file, 'r', encoding='utf-8') as f:
                        paper_content = f.read()
        
        # 构建输入输出对
        if paper_content:
            input_text = f"Paper Title: {title}\n\nPaper Content: {paper_content[:4000]}\n\nWrite a comprehensive review for this paper."
        else:
            input_text = f"Paper Title: {title}\n\nWrite a comprehensive review for this paper."
            
        output_text = item['review_text']  # 或 aggregated_review
        
        processed_data.append({
            "input": input_text,
            "output": output_text
        })
    
    return processed_data

def train_model(model, dataset_path):
    """训练模型"""
    # 准备数据集
    dataset = prepare_dataset(dataset_path)
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="paged_adamw_32bit"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset
    )
    
    trainer.train()
