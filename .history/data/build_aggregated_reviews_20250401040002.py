import os
import json
import glob
from pathlib import Path

def aggregate_txt_reviews(txt_dir, output_path):
    """
    从txt文件聚合评审数据并保存为JSON格式
    :param txt_dir: 包含txt文件的目录路径
    :param output_path: 输出JSON文件路径
    """
    aggregated_reviews = []
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 递归查找所有txt文件
    for txt_file in glob.glob(os.path.join(txt_dir, '**', '*.txt'), recursive=True):
        try:
            # 获取文件名作为paper_id
            paper_id = Path(txt_file).stem
            
            # 读取文件内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 简单解析内容 - 假设第一行是标题，其余是评审内容
            lines = content.split('\n')
            title = lines[0].strip() if lines else "Untitled"
            review_content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else content
            
            # 构建评审条目
            review_entry = {
                "id": paper_id,
                "title": title,
                "conference": "",  # 可根据需要从文件名/路径提取
                "year": "",       # 可根据需要从文件名/路径提取
                "aggregated_review": review_content
            }
            
            aggregated_reviews.append(review_entry)
            
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {e}")
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_reviews, f, ensure_ascii=False, indent=2)
    
    print(f"成功聚合 {len(aggregated_reviews)} 条评审数据到 {output_path}")

if __name__ == "__main__":
    # 配置路径
    base_dir = "/Users/arist/Documents/llm-review-sys"
    txt_dir = os.path.join(base_dir, "data", "extracted_texts")  # 存放txt文件的目录
    output_path = os.path.join(base_dir, "data", "aggregated_reviews", "aggregated_reviews.json")
    
    # 执行聚合
    aggregate_txt_reviews(txt_dir, output_path)