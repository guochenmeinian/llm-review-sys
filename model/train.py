import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from qlora import setup_qlora_model, prepare_dataset

def main():
    # 配置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    review_dataset_path = os.path.join(base_dir, "data/openreview/openreview_dataset.json")
    parsed_texts_dir = os.path.join(base_dir, "data/parsed_texts")
    output_dir = os.path.join(base_dir, "model/results")
    
    # 设置模型
    model = setup_qlora_model()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 准备数据集
    raw_data = prepare_dataset(review_dataset_path, parsed_texts_dir)
    dataset = Dataset.from_list(raw_data)
    
    # 数据预处理
    def preprocess_function(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        
        # 构建完整的提示和回答格式
        model_inputs = tokenizer(
            inputs, 
            max_length=4096,
            padding="max_length",
            truncation=True
        )
        
        # 准备标签
        labels = tokenizer(
            outputs,
            max_length=1024,
            padding="max_length",
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # 处理数据集
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # 减小批量大小以适应更长的输入
        gradient_accumulation_steps=16, # 增加梯度累积步数
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
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
    trainer.save_model(os.path.join(output_dir, "paper_review_model"))

if __name__ == "__main__":
    main()