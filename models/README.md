## Model Descriptions

- `full_context_qlora_model/`: This model was fine-tuned using QLoRA on full-paper inputs (~18,000 tokens) with an H100 GPU. We trained for **2 epochs**. It is suitable for tasks requiring full document understanding.
- `sliding_window_qlora_model/`: This model was fine-tuned with QLoRA using sliding-window inputs (8192 tokens per segment with overlap) on a 4090 GPU. It was also trained for **2 epochs**. This setup is useful when working with GPUs that have limited memory.
- `dpo_base_model/`: This model is further aligned using DPO (Direct Preference Optimization) on top of the full_context_qlora_model. The `chosen` responses are manually aggregated high-quality reviews, while the `rejected` responses come from the original LLaMA3.1 outputs.
  

