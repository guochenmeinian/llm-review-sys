# Automated LLM-Based Peer Review System

This project uses LLM to auto-reviews the research paper.

- We crawled papers from openreview with top venues papers (e.g. ICML, NIPS, ICLR, CVPR) along with their reviews. The dataset is published in Huggingface and made public: [Link](https://huggingface.co/datasets/guochenmeinian/openreview)

- We used [Nougat-OCR](https://github.com/facebookresearch/nougat) to parse the PDF. Here's a [usage](https://github.com/ad17171717/YouTube-Tutorials/blob/main/Machine%20Learning%20with%20Python/Optical_Character_Recognition_(OCR)_with_Meta's_Nougat!.ipynb) guide I found. With the raw data, We use LLM APIs to format/merge reviews to prepare the dataset for training.

- (To-Do): We applied QLora and Prefix Insertion to finetune on LLama3-8b-instruct-8k model with our dataset.

- (To-Do): We further train the model with DPO technique to ensure accuracy.

(Ideally) The matching rate between the reviews generated by our model and the manual reviews surpassed the effect of GPT-4 reviewers, improving the accuracy of automated review.


---

start with setup the env:
```
conda env create -f environment.yml
conda activate openreview
```
if you changed the env, do this:
`conda env export > environment.yml`

---



