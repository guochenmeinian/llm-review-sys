from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PrefixTuningConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def setup_prefix_model(model_name="meta-llama/Llama-3-8b-instruct"):
    """设置带有Prefix Tuning的模型"""
    # 加载基础模型
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
    
    model = prepare_model_for_kbit_training(model)
    
    # 配置Prefix Tuning
    peft_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=20,
        prefix_projection=True,
        token_dim=model.config.hidden_size,
        num_layers=model.config.num_hidden_layers,
        fan_in_fan_out=False
    )
    
    return get_peft_model(model, peft_config)