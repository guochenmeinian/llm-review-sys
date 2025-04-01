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
            paper_id = Path(txt_file).stem
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 解析文件内容 - 假设格式为：
            # 标题
            # 会议: 会议名称
            # 年份: 年份
            # 评审内容...
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            title = lines[0] if lines else "Untitled"
            conference = ""
            year = ""
            review_content = ""
            
            # 解析会议和年份信息
            for line in lines[1:]:
                if line.lower().startswith("会议:"):
                    conference = line.split(":", 1)[1].strip()
                elif line.lower().startswith("年份:"):
                    year = line.split(":", 1)[1].strip()
                else:
                    review_content += line + "\n"
            
            review_entry = {
                "id": paper_id,
                "title": title,
                "conference": conference,
                "year": year,
                "aggregated_review": review_content.strip()
            }
            
            aggregated_reviews.append(review_entry)
            
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