import os
import json
from collections import defaultdict
from datetime import datetime

def process_reviews(input_dir, output_dir):
    """
    处理所有会议数据并生成 **单个** Hugging Face 兼容的 `openreview_dataset.json`
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
        if conference == "dataset" or not os.path.isdir(os.path.join(input_dir, conference)):
            continue

        results_path = os.path.join(input_dir, conference, "results.json")
        if not os.path.exists(results_path):
            continue

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for venue in data:
                process_venue(venue, merged_data, stats)

    # 保存为 **一个** JSON 文件
    output_file = os.path.join(output_dir, "openreview_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    # 生成数据统计信息
    generate_summary(output_dir, stats, len(merged_data))
    
    print(f"✅ 处理完成，共 {len(merged_data)} 条数据，已保存到 {output_file}")

def process_venue(venue, merged_data, stats):
    """处理单个会议的评审数据"""
    for paper in venue["papers"]:
        # 更新年份统计
        if paper["year"]:
            stats["min_year"] = min(stats["min_year"], paper["year"])
            stats["max_year"] = max(stats["max_year"], paper["year"])
        
        # 生成基础信息
        base_info = {
            "title": paper.get("title", "Unknown"),
            "conference": paper.get("conference", "Unknown"),
            "pdf_url": paper.get("pdf_url", ""),
            "year": paper.get("year", 0)  # 确保 year 是 int 类型
        }
        
        # 处理每个评审
        for review in paper["reviews"]:
            entry = create_entry(base_info, review)
            merged_data.append(entry)
            
            # 更新统计
            stats["conferences"][paper["conference"]] += 1
            stats["review_types"][review["type"].lower()] += 1

def create_entry(base_info, review):
    """创建单个评审条目，并展开 content 结构"""
    try:
        # 确保 content 是 JSON 格式
        content = review.get("content", {})
        if isinstance(content, str):
            content = json.loads(content)  # 解析 JSON 字符串
        
        structured_fields = {
            "content_title": content.get("title", {}).get("value", "No title"),
            "rating": content.get("rating", {}).get("value", -1),
            "confidence": content.get("confidence", {}).get("value", -1),
            "recommendation": content.get("recommendation", {}).get("value", "No recommendation"),
            "review_type": review.get("type", "unknown").lower(),
        }

        review_type = structured_fields["review_type"]

        # 处理 review_text
        if review_type == "official_review":
            excluded_keys = {"title", "rating", "confidence", "recommendation"}
            merged_review_text = "\n".join(
                f"{field.replace('_', ' ').title()}: {content[field].get('value', '')}" 
                for field in content.keys() if field not in excluded_keys and "value" in content[field]
            ).strip()
            structured_fields["review_text"] = merged_review_text if merged_review_text else "No review"
        elif review_type == "meta_review":
            # For Meta_Review, just extract `metareview`
            structured_fields["review_text"] = content.get("metareview", {}).get("value", "No review")
        elif review_type == "official_comment":
            structured_fields["review_text"] = content.get("comment", {}).get("value", "No comment")

        return {**base_info, **structured_fields}

    except Exception as e:
        print(f"⚠️ 解析 review 失败: {e}")
        return None


def safe_number(value):
    """确保评分值是 int 或 float，防止 NaN 或 'N/A' 影响 Hugging Face 加载"""
    try:
        return int(value) if float(value).is_integer() else float(value)
    except (ValueError, TypeError):
        return -1  # 默认值

def generate_summary(output_dir, stats, total_entries):
    """生成数据集统计信息"""
    summary = {
        "total_reviews": total_entries,
        "time_span": f"{stats['min_year']}-{stats['max_year']}",
        "conference_distribution": dict(stats["conferences"]),
        "review_type_distribution": dict(stats["review_types"]),
        "last_updated": datetime.now().isoformat()
    }
    
    # with open(os.path.join(output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
    #   json.dump(summary, f, indent=2, ensure_ascii=False)
    print(summary)

if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(__file__), "openreview")
    output_dir = os.path.join(os.path.dirname(__file__), "openreview")
    process_reviews(input_dir=input_dir, output_dir=output_dir)
