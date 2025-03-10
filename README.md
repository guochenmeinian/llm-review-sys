# Automated LLM-Based Peer Review System

start with setup the env:
```
conda env create -f environment.yml
conda activate paper_finetune
```
if you changed the env, do this:
`conda env export > environment.yml`


we crawled papers from top venues (i.e. icml-neurips-iclr paers). dataset are published in huggingface and made public: https://huggingface.co/datasets/guochenmeinian/icml-neurips-iclr-review-dataset

this project uses LLM to auto-reviews the research paper.