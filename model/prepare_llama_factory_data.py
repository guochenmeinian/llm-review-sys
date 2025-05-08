import os
import json
from tqdm import tqdm

def format_for_llama_factory(dataset_path, output_path, max_input_length=50000, min_input_length=10000):
    """将数据集格式化为Llama Factory所需的格式，并过滤掉过长或过短的输入"""
    # 加载数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为Llama Factory格式
    llama_factory_data = []
    filtered_long_count = 0
    filtered_short_count = 0
    
    print(f"正在处理数据集，共 {len(data)} 条记录...")
    
    for item in tqdm(data, desc="格式化并过滤数据"):
        # 获取平均评分和置信度
        rating = item.get('avg_rating', 0)
        confidence = item.get('avg_confidence', 0)
        
        # 构建提示和回复
        prompt = f"""Paper Details:
- Title: {item['title']}

- Conference: {item['conference']} {item['year']}

- Content: {item['paper_content']}"""

        # 构建标准化的输出格式
        response = f"{item['aggregated_review']}\n### Rating\nOverall Quality: {rating:.1f}\nReview Confidence: {confidence:.1f}"
        
        llama_factory_item = {
            "instruction": f"""You are an academic paper reviewer. Please write a structured review of the following paper, using only the four sections below. Keep your response concise, objective, and focused. Do not include any content beyond these four sections:

### Key Points
Summarize the main contributions and key ideas of the paper.

### Strengths and Weaknesses
Strengths:
- List the paper's major strengths and significant contributions
Weaknesses:
- Identify areas that need improvement

### Suggestions for Improvement
Provide specific recommendations for enhancing the paper.

### Rating
Overall Quality: (1-10)
Review Confidence: (1-5)
""",
            "input": prompt,
            "output": response
        }
        
        # 检查输入长度
        input_length = len(prompt)
        if input_length > max_input_length:
            filtered_long_count += 1
            continue
        elif input_length < min_input_length:
            filtered_short_count += 1
            continue
        
        llama_factory_data.append(llama_factory_item)
    
    # 保存Llama Factory格式数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
       json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)
    
    print(f"已创建Llama Factory格式数据集，包含 {len(llama_factory_data)} 条记录")
    if filtered_long_count > 0:
        print(f"已过滤 {filtered_long_count} 条过长输入 (超过 {max_input_length} 字符)")
    if filtered_short_count > 0:
        print(f"已过滤 {filtered_short_count} 条过短输入 (少于 {min_input_length} 字符)")
    
    return llama_factory_data

def create_train_dpo_test_split(dataset_path, train_path, dpo_path, test_path, train_ratio=0.75, dpo_ratio=0.23, test_ratio=0.02):
    """按 75/23/2 划分训练集（QLoRA）、DPO集、测试集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    test_size = max(1, int(total * test_ratio))
    dpo_size = max(1, int(total * dpo_ratio))
    train_size = total - dpo_size - test_size

    # 顺序分配（如需随机可打乱）
    train_data = data[:train_size]
    dpo_data = data[train_size:train_size + dpo_size]
    test_data = data[train_size + dpo_size:]

    # 保存
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(dpo_path), exist_ok=True)
    with open(dpo_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"已创建训练集 ({len(train_data)} 条记录)、DPO集 ({len(dpo_data)} 条记录)、测试集 ({len(test_data)} 条记录)")

    return train_data, dpo_data, test_data


def filter_long_inputs(input_path, output_path, max_length=150000):
    """过滤掉输入长度超过指定长度的数据集条目"""
    print(f"正在加载数据集: {input_path}")
    
    # 加载数据集
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    print(f"原始数据集包含 {original_count} 条记录")
    
    # 过滤长输入
    filtered_data = []
    removed_count = 0
    
    for item in tqdm(data, desc="过滤长输入"):
        # 计算input字段的长度
        input_length = len(item.get("input", ""))
        
        # 如果长度在允许范围内，保留该条目
        if input_length <= max_length:
            filtered_data.append(item)
        else:
            removed_count += 1
    
    # 保存过滤后的数据集
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n过滤完成!")
    print(f"移除了 {removed_count} 条长输入记录 ({removed_count/original_count:.2%})")
    print(f"保留了 {len(filtered_data)} 条记录")
    print(f"过滤后的数据集已保存到: {output_path}")
    
    return filtered_data

if __name__ == "__main__":
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dataset_path = os.path.join(base_dir, "data", "paper_review_dataset", "paper_review_dataset.json")
    output_dir = os.path.join(base_dir, "model", "llama_factory_dataset")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径
    llama_factory_output = os.path.join(output_dir, "llama_factory_format.json")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_dataset_path):
        print(f"输入数据集文件不存在: {input_dataset_path}")
        print("请先运行 create_llama_factory_dataset.py 创建数据集")
        exit(1)
    
    # 创建Llama Factory格式数据
    print("创建Llama Factory格式数据...")
    llama_factory_data = format_for_llama_factory(input_dataset_path, llama_factory_output)
    
    # 自动创建训练集、验证集和测试集分割
    qlora_output = os.path.join(output_dir, "llama_factory_qlora.json")
    dpo_output = os.path.join(output_dir, "llama_factory_dpo.json")
    eval_output = os.path.join(output_dir, "llama_factory_eval.json")

    print("创建训练集、DPO集和测试集分割...")
    qlora_data, dpo_data, eval_data = create_train_dpo_test_split(
        llama_factory_output, qlora_output, dpo_output, eval_output
    )
    
    # print(f"Llama Factory格式数据已保存到: {llama_factory_output}")
    print(f"QLora训练集已保存到: {qlora_output} ({len(qlora_data)} 条记录)")
    print(f"DPO训练集已保存到: {dpo_output} ({len(dpo_data)} 条记录)")
    print(f"测试集已保存到: {eval_output} ({len(eval_data)} 条记录)")
    
    print("转换完成！")