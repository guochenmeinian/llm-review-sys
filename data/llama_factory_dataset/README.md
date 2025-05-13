---
language:
- en
license: apache-2.0
datasets:
- openreview
tags:
- paper-review
- academic-review
- nlp
- openreview
configs:
  - config_name: qlora_train
    data_files: qlora_train.json
  - config_name: qlora_validation
    data_files: qlora_validation.json
  - config_name: dpo_original
    data_files: dpo.json
  - config_name: dpo_base
    data_files: dpo_base_model.json
  - config_name: eval
    data_files: eval.json
---

# 📘 Dataset Overview
This dataset was developed as part of a course project for [CSCI-GA 2565: Machine Learning](https://cs.nyu.edu/courses/spring25/CSCI-GA.2565-001/) at NYU. Our objective was to build a training pipeline that enables LLMs to perform structured academic paper reviewing.

## 🛠 Data Collection & Preprocessing

- **Paper Content**:  
  Raw papers were collected from [OpenReview](https://openreview.net) using their official API. The PDFs were converted into Markdown format using [Nougat OCR](https://github.com/facebookresearch/nougat), followed by a custom cleaning pipeline involving regular expressions to remove OCR artifacts, redundant symbols, and formatting inconsistencies.

- **Review Aggregation**:  
  OpenReview often includes multiple reviews per paper. We used multi-step prompting with GPT-4 to generate an **aggregated structured review**, consolidating multiple perspectives into a single coherent evaluation covering key points, strengths/weaknesses, and suggestions.

> ⚠️ **Note**: While the dataset quality is sufficient for research and experimentation, it may still contain noisy or imperfect samples due to the automated preprocessing and OCR limitations.

## Upload
`huggingface-cli upload guochenmeinian/openreview_dataset . --repo-type=dataset`

