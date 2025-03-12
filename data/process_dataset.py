import os
import json
from collections import defaultdict
from datetime import datetime

def process_reviews(input_dir, output_dir, chunk_size=1000):
    """
    处理所有会议数据并生成符合HuggingFace格式的文件
    参数：
        chunk_size: 每个JSON文件包含的最大评审条目数（默认1000）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    merged_data = []
    stats = {
        "conferences": defaultdict(int),
        "review_types": defaultdict(int),
        "min_year": datetime.now().year,
        "max_year": 2000
    }

    # 遍历会议目录
    for conference in os.listdir(input_dir):
        # 跳过非目录或数据集目录
        if conference == "dataset" or not os.path.isdir(os.path.join(input_dir, conference)):
            continue

        # 查找并处理 results.json 文件
        results_path = os.path.join(input_dir, conference, "results.json")
        if not os.path.exists(results_path):
            continue

        with open(results_path, "r") as f:
            data = json.load(f)
            for venue in data:
                process_venue(venue, merged_data, stats)

    # 保存为单个文件
    output_file = os.path.join(output_dir, "openreview_dataset.json")
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)
    
    # 生成总结文件
    generate_summary(output_dir, stats, len(merged_data))
    
    print(f"✅ 成功处理 {len(merged_data)} 条数据，保存到 {output_file}")

def process_venue(venue, merged_data, stats):
    """处理单个venue的数据"""
    for paper in venue["papers"]:
        # 更新年份统计
        if paper["year"]:
            stats["min_year"] = min(stats["min_year"], paper["year"])
            stats["max_year"] = max(stats["max_year"], paper["year"])
        
        # 生成基础信息
        base_info = {
            "title": paper["title"],
            "conference": paper["conference"],
            "pdf_url": paper["pdf_url"],
            "year": paper["year"]
        }
        
        # 处理每个评审
        for review in paper["reviews"]:
            entry = create_entry(base_info, review)
            merged_data.append(entry)
            
            # 更新统计
            stats["conferences"][paper["conference"]] += 1
            stats["review_types"][review["type"].lower()] += 1

def create_entry(base_info, review):
    """创建单个评审条目"""
    return {
        **base_info,
        "review_type": review["type"].lower(),
        "content": review["content"].get("comment", ""),
        "ratings": review["ratings"].get("rating", "N/A"),
        "confidence": review["ratings"].get("confidence", "N/A")
    }

def save_chunks(data, chunk_size, output_dir, stats):
    """分块保存数据"""
    total_files = (len(data) - 1) // chunk_size + 1
    
    for i in range(total_files):
        chunk = data[i*chunk_size : (i+1)*chunk_size]
        filename = f"reviews_{i+1:03d}.json"
        
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(chunk, f, indent=2)

def generate_summary(output_dir, stats, total_entries):
    """生成数据集总结文件"""
    summary = {
        "total_reviews": total_entries,
        "time_span": f"{stats['min_year']}-{stats['max_year']}",
        "conference_distribution": dict(stats["conferences"]),
        "review_type_distribution": dict(stats["review_types"]),
        "last_updated": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(__file__), "openreview")
    output_dir = os.path.join(input_dir, "dataset")
    process_reviews(input_dir=input_dir, output_dir=output_dir)