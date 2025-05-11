### 🏋️ Model Training Overview

We explored how to train LLaMA-3.1 models using both the official LLaMA-Factory and our own custom training pipeline to understand the full stack of training large language models — from prompt formatting to QLoRA fine-tuning and DPO alignment.

### Strategy:

We encountered memory constraints when training LLaMA-3.1 with long contexts (e.g. 18,000 tokens) on AutoDL. Even with 4-bit QLoRA and gradient checkpointing, a single forward pass failed on 4090s (24GB VRAM).

#### Our solutions:

- **H100 for full-context training (18000 tokens):**
We rented a H100 instance on [Run Pod](https://www.runpod.io/) to handle the full-sequence fine-tuning in one go.

- **Sliding window strategy on 4090s:**
For limited GPU memory, we trained a second QLoRA model by chunking long papers into overlapping windows (i.e., 8000 tokens per chunk with overlaps), so each pass fits into memory.


### Summary
| Method               | Device            | Description                               |
| -------------------- | ----------------- | ----------------------------------------- |
| Full-context QLoRA   | 1× H100           | Trained with 18000-token full paper input |
| Sliding-window QLoRA | 1× 4090 (x5 GPUs) | Trained on segments of the same input     |
| DPO                  | On top of QLoRA   | Fine-tuned using chosen/rejected reviews  |

