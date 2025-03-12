from trl import DPOTrainer

def dpo_training(model, tokenizer, dataset):
    """进行DPO训练"""
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            num_train_epochs=2,
            output_dir="./dpo_results"
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1
    )
    
    dpo_trainer.train()