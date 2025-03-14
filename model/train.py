from transformers import AutoTokenizer, TrainingArguments, Trainer
import json
from datasets import Dataset
import torch
from qlora import setup_qlora_model
from prefix_tuning import setup_prefix_model

def prepare_dataset(data_path):
    """准备训练数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建训练样本
    processed_data = []
    for item in data:
        processed_data.append({
            "input": f"Paper Title: {item['title']}\n\nWrite a comprehensive review for this paper.",
            "output": item['review_text']
        })
    
    return Dataset.from_list(processed_data)

def train(
    model_type="qlora",
    model_name="meta-llama/Llama-3-8b-instruct",
    dataset_path="/Users/arist/Documents/llm-review-sys/data/openreview_dataset.json",
    output_dir="./results",
    batch_size=4,
    epochs=3,
    learning_rate=2e-4
):
    """训练模型"""
    # 选择模型类型
    if model_type.lower() == "qlora":
        model = setup_qlora_model(model_name)
    elif model_type.lower() == "prefix":
        model = setup_prefix_model(model_name)
    else:
        raise ValueError("模型类型必须是 'qlora' 或 'prefix'")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 准备数据集
    dataset = prepare_dataset(dataset_path)
    
    def preprocess_function(examples):
        # 将输入和输出组合成完整的文本
        texts = [f"{input_text}\n{output_text}" for input_text, output_text in 
                zip(examples["input"], examples["output"])]
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
    
    # 对数据集进行预处理
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="paged_adamw_32bit"
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model(f"{output_dir}/{model_type}_final")

# if __name__ == "__main__":
#     # QLora训练示例
#     train(
#         model_type="qlora",
#         output_dir="./results/qlora"
#     )
    
#     # Prefix Tuning训练示例
#     train(
#         model_type="prefix",
#         output_dir="./results/prefix"
#     )