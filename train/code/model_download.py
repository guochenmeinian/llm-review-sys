import torch
import os
from modelscope import snapshot_download, AutoModel, AutoTokenizer
model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct',
cache_dir='/root/autodl-tmp', revision='master')