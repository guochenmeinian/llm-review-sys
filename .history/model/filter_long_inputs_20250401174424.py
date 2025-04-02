import os
import json
from tqdm import tqdm

def filter_long_inputs(input_path, output_path, max_length=150000):
    """
    过滤掉输入长度超过指定长度的数据集条目
    
    参数:
        input_path: 输入数据集路径
        output_path: 输出数据集路径
        max_length: 最大允许的输入长度，默认为150000字符
    """
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

def process_all_datasets(base_dir):
    """处理llama_factory_dataset目录下的所有数据集文件"""
    dataset_dir = os.path.join(base_dir, "model", "llama_factory_dataset")
    
    # 查找所有json文件
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        input_path = os.path.join(dataset_dir, json_file)
        # 创建输出文件名，添加_filtered后缀
        filename, ext = os.path.splitext(json_file)
        output_file = f"{filename}_filtered{ext}"
        output_path = os.path.join(dataset_dir, output_file)
        
        print(f"\n处理文件: {json_file}")
        filter_long_inputs(input_path, output_path)

if __name__ == "__main__":
    # 设置基础目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 处理单个文件
    # input_path = os.path.join(base_dir, "model", "llama_factory_dataset", "llama_factory_format.json")
    # output_path = os.path.join(base_dir, "model", "llama_factory_dataset", "llama_factory_format_filtered.json")
    # filter_long_inputs(input_path, output_path)
    
    # 处理所有数据集文件
    process_all_datasets(base_dir)