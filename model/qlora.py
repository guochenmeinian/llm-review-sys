from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
from datasets import load_dataset, Dataset
import json
import os

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

def load_llama_factory_dataset(dataset_path):
    """加载Llama Factory格式的数据集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为HuggingFace Dataset格式
    processed_data = []
    for item in data:
        processed_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        })
    
    return Dataset.from_list(processed_data)

def preprocess_function(examples, tokenizer, max_input_length=4096, max_output_length=1024):
    """预处理函数，将数据转换为模型输入格式"""
    # 组合指令和输入
    if examples["input"]:
        inputs = [f"{examples['instruction']}\n\n{inp}" for inp in examples["input"]]
    else:
        inputs = [examples["instruction"]]
    
    outputs = examples["output"]
    
    # 对输入进行编码
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )
    
    # 对输出进行编码
    labels = tokenizer(
        outputs,
        max_length=max_output_length,
        padding="max_length",
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_qlora(dataset_path, output_dir="./results/qlora", model_name="meta-llama/Llama-3-8b-instruct", epochs=3):
    """使用QLoRA方法训练模型"""
    # 设置模型
    model = setup_qlora_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = load_llama_factory_dataset(dataset_path)
    
    # 数据预处理
    def process_batch(batch):
        return preprocess_function(batch, tokenizer)
    
    tokenized_dataset = dataset.map(
        process_batch,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_32bit"
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    print(f"模型训练完成，已保存到 {os.path.join(output_dir, 'final_model')}")

if __name__ == "__main__":
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "model", "llama_factory_dataset", "llama_factory_format.json")
    output_dir = os.path.join(base_dir, "model", "results", "qlora")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 开始训练
    print("开始QLoRA训练...")
    train_qlora(dataset_path, output_dir)
