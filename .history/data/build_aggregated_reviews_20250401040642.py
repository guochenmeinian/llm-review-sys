import os
import json
import glob
from pathlib import Path

def aggregate_txt_reviews(txt_dir, output_path):
    aggregated_reviews = []
    
    # 添加路径检查
    if not os.path.exists(txt_dir):
        print(f"错误：目录不存在 - {txt_dir}")
        return []
    
    print(f"正在从目录扫描txt文件: {txt_dir}")
    
    # 递归查找所有txt文件
    txt_files = list(glob.glob(os.path.join(txt_dir, '**', '*.txt'), recursive=True))
    print(f"找到 {len(txt_files)} 个txt文件")
    
    for txt_file in txt_files:
        try:
            # 从文件名获取paper_id (去掉年份部分)
            filename = Path(txt_file).stem
            paper_id = filename.split('_')[0]
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 初始化变量
            review_data = {
                "id": paper_id,
                "title": "",
                "conference": "",
                "year": "",
                "number_of_reviews": 0,
                "original_ratings": [],
                "original_confidences": [],
                "aggregated_review": ""
            }
            
            # 按行解析内容
            current_section = None
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # 解析元数据部分
                if line.startswith("ID:"):
                    review_data["id"] = line.split(":", 1)[1].strip()
                elif line.startswith("Title:"):
                    review_data["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("Conference:"):
                    review_data["conference"] = line.split(":", 1)[1].strip()
                elif line.startswith("Year:"):
                    review_data["year"] = line.split(":", 1)[1].strip()
                elif line.startswith("Number of Reviews:"):
                    review_data["number_of_reviews"] = int(line.split(":", 1)[1].strip())
                elif line.startswith("Original Ratings:"):
                    ratings = line.split(":", 1)[1].strip()
                    review_data["original_ratings"] = [int(r) for r in ratings.split(",") if r.strip().isdigit()]
                elif line.startswith("Original Confidences:"):
                    confidences = line.split(":", 1)[1].strip()
                    review_data["original_confidences"] = [int(c) for c in confidences.split(",") if c.strip().isdigit()]
                elif line.startswith("Aggregated Review:"):
                    current_section = "review"
                elif current_section == "review":
                    review_data["aggregated_review"] += line + "\n"
            
            # 添加到结果列表
            aggregated_reviews.append(review_data)
            
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {e}")
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_reviews, f, ensure_ascii=False, indent=2)
    
    print(f"成功聚合 {len(aggregated_reviews)} 条评审数据到 {output_path}")

if __name__ == "__main__":
    # 修改为使用相对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_dir = os.path.join(base_dir, "data", "aggregated_reviews")
    output_path = os.path.join(base_dir, "data", "aggregated_reviews", "aggregated_reviews.json")
    
    # 执行聚合
    aggregate_txt_reviews(txt_dir, output_path)