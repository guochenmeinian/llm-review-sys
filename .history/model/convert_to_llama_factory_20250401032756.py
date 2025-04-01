import os
import json

def format_for_llama_factory(dataset_path, output_path):
    """将数据集格式化为Llama Factory所需的格式"""
    # 加载数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为Llama Factory格式
    llama_factory_data = []
    
    for item in data:
        # 构建提示和回复
        prompt = f"Title: {item['title']}\n\nConference: {item['conference']} {item['year']}\n\n{item['paper_content']}\n\nPlease provide a comprehensive review of this paper."
        response = item['aggregated_review']
        
        llama_factory_item = {
            "instruction": "You are a helpful academic paper reviewer. Please provide a comprehensive review of the following paper.",
            "input": prompt,
            "output": response
        }
        
        llama_factory_data.append(llama_factory_item)
    
    # 保存Llama Factory格式数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)
    
    print(f"已创建Llama Factory格式数据集，包含 {len(llama_factory_data)} 条记录")
    return llama_factory_data

def create_train_val_split(dataset_path, train_path, val_path, val_ratio=0.1):
    """创建训练集和验证集分割"""
    # 加载完整数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算验证集大小
    val_size = max(1, int(len(data) * val_ratio))
    
    # 分割数据
    train_data = data[:-val_size] if val_size > 0 else data
    val_data = data[-val_size:] if val_size > 0 else []
    
    # 保存训练集
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"已创建训练集 ({len(train_data)} 条记录) 和验证集 ({len(val_data)} 条记录)")
    
    return train_data, val_data

def create_train_val_test_split(dataset_path, train_path, val_path, test_path, val_ratio=0.1, test_ratio=0.1):
    """创建训练集、验证集和测试集分割"""
    # 加载完整数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算验证集和测试集大小
    test_size = max(1, int(len(data) * test_ratio))
    val_size = max(1, int(len(data) * val_ratio))
    
    # 分割数据
    test_data = data[-test_size:] if test_size > 0 else []
    val_data = data[-(test_size+val_size):-test_size] if val_size > 0 else []
    train_data = data[:-(test_size+val_size)] if (test_size+val_size) > 0 else data
    
    # 保存训练集
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    # 保存测试集
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"已创建训练集 ({len(train_data)} 条记录)、验证集 ({len(val_data)} 条记录)和测试集 ({len(test_data)} 条记录)")
    
    return train_data, val_data, test_data

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
    
    # 询问是否需要创建数据集分割
    create_split = input("是否需要创建数据集分割？(y/N): ").strip().lower()
    if create_split == 'y':
        split_type = input("选择分割类型：1. 训练集和验证集 2. 训练集和测试集 3. 训练集、验证集和测试集 (默认3): ").strip()
        
        if split_type == '1':
            train_output = os.path.join(output_dir, "llama_factory_train.json")
            val_output = os.path.join(output_dir, "llama_factory_val.json")
            
            # 创建训练集和验证集分割
            print("创建训练集和验证集分割...")
            train_data, val_data = create_train_val_split(llama_factory_output, train_output, val_output)
            
            print(f"Llama Factory格式数据已保存到: {llama_factory_output}")
            print(f"训练集已保存到: {train_output}")
            print(f"验证集已保存到: {val_output}")
        elif split_type == '2':
            train_output = os.path.join(output_dir, "llama_factory_train.json")
            test_output = os.path.join(output_dir, "llama_factory_test.json")
            
            # 创建训练集和测试集分割
            print("创建训练集和测试集分割...")
            train_data, test_data = create_train_test_split(llama_factory_output, train_output, test_output)
            
            print(f"Llama Factory格式数据已保存到: {llama_factory_output}")
            print(f"训练集已保存到: {train_output}")
            print(f"测试集已保存到: {test_output}")
        else:  # 默认为3或其他输入
            train_output = os.path.join(output_dir, "llama_factory_train.json")
            val_output = os.path.join(output_dir, "llama_factory_val.json")
            test_output = os.path.join(output_dir, "llama_factory_test.json")
            
            # 创建训练集、验证集和测试集分割
            print("创建训练集、验证集和测试集分割...")
            train_data, val_data, test_data = create_train_val_test_split(
                llama_factory_output, train_output, val_output, test_output
            )
            
            print(f"Llama Factory格式数据已保存到: {llama_factory_output}")
            print(f"训练集已保存到: {train_output}")
            print(f"验证集已保存到: {val_output}")
            print(f"测试集已保存到: {test_output}")
    else:
        print(f"Llama Factory格式数据已保存到: {llama_factory_output}")
    
    print("转换完成！")