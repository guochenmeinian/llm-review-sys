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
    invalid_papers = []
    
    for item in data:
        if item.get('avg_rating', -1) == -1 or item.get('avg_confidence', -1) == -1:
            invalid_papers.append({
                'id': item.get('id'),
                'title': item.get('title'),
                'conference': item.get('conference'),
                'year': item.get('year'),
                'avg_rating': item.get('avg_rating'),
                'avg_confidence': item.get('avg_confidence'),
                'original_ratings': item.get('original_ratings', []),
                'original_confidences': item.get('original_confidences', [])
            })
    
    print(f"数据集总论文数: {total_papers}")
    print(f"无效评分论文数: {len(invalid_papers)} ({len(invalid_papers)/total_papers*100:.2f}%)")
    
    # 分析无效原因
    empty_ratings = sum(1 for p in invalid_papers if not p['original_ratings'])
    empty_confidences = sum(1 for p in invalid_papers if not p['original_confidences'])
    
    print("\n无效原因分析:")
    print(f"完全没有原始评分的论文数: {empty_ratings}")
    print(f"完全没有原始置信度的论文数: {empty_confidences}")
    
    print("\n按会议分布:")
    conference_stats = {}
    for p in invalid_papers:
        conf = p.get('conference', 'unknown')
        conference_stats[conf] = conference_stats.get(conf, 0) + 1
    for conf, count in conference_stats.items():
        print(f"{conf}: {count}篇")
    
    print("\n示例无效数据:")
    for paper in invalid_papers[:3]:  # 显示前3个示例
        print("-" * 50)
        print(f"ID: {paper['id']}")
        print(f"标题: {paper['title']}")
        print(f"会议: {paper['conference']}")
        print(f"年份: {paper['year']}")
        print(f"平均评分: {paper['avg_rating']}")
        print(f"平均置信度: {paper['avg_confidence']}")
        print(f"原始评分: {paper['original_ratings']}")
        print(f"原始置信度: {paper['original_confidences']}")

if __name__ == "__main__":
    analyze_ratings()