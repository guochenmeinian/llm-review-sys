import json
import os

def analyze_ratings():
    dataset_path = '/Users/arist/Documents/llm-review-sys/data/paper_review_dataset/paper_review_dataset.json'
    
    if not os.path.exists(dataset_path):
        print("数据集文件不存在")
        return
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_papers = len(data)
    invalid_rating_count = sum(1 for item in data if item.get('avg_rating', -1) == -1)
    invalid_confidence_count = sum(1 for item in data if item.get('avg_confidence', -1) == -1)
    
    print(f"数据集总论文数: {total_papers}")
    print(f"无效平均评分(avg_rating=-1)数量: {invalid_rating_count} ({(invalid_rating_count/total_papers)*100:.2f}%)")
    print(f"无效平均置信度(avg_confidence=-1)数量: {invalid_confidence_count} ({(invalid_confidence_count/total_papers)*100:.2f}%)")
    
    # 输出一些示例
    print("\n以下是部分无效评分的论文:")
    for item in data:
        if item.get('avg_rating', -1) == -1 or item.get('avg_confidence', -1) == -1:
            print(f"ID: {item.get('id')}")
            print(f"标题: {item.get('title')}")
            print(f"平均评分: {item.get('avg_rating')}")
            print(f"平均置信度: {item.get('avg_confidence')}")
            print(f"原始评分: {item.get('original_ratings', [])}")
            print(f"原始置信度: {item.get('original_confidences', [])}")
            print("-" * 50)
            break  # 只显示第一个示例

if __name__ == "__main__":
    analyze_ratings()