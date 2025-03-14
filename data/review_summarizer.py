from openai import OpenAI
import json
import pandas as pd
import os

class ReviewAggregator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def aggregate_reviews(self, reviews):
        """使用统一英文prompt处理单条或多条review"""
        prompt = f"""
        Please generate a concise summary based on the following review(s):
        1. Key points
        2. Strengths and weaknesses
        3. Suggestions for improvement
        
        Requirements:
        - Maintain the core viewpoints of the original review(s)
        - Do not add new information
        - Use concise language
        
        Review(s):
        {json.dumps(reviews, ensure_ascii=False)}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content

    def process_openreview_dataset(self, dataset_path):
        """处理数据集并聚合评审意见"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 按论文标题组织评审
        paper_reviews = {}
        
        for item in data:
            # 提取标题和评审内容
            title = item.get('title')
            review_text = item.get('review_text')
            
            if title and review_text:
                # 如果标题不存在，创建新条目
                if title not in paper_reviews:
                    paper_reviews[title] = []
                
                # 构建结构化的评审信息
                review_info = {
                    'review_text': review_text,
                    'rating': item.get('ratings'),
                    'confidence': item.get('confidence'),
                    'conference': item.get('conference'),
                    'year': item.get('year')
                }
                
                paper_reviews[title].append(review_info)
        
        # 聚合每篇论文的评审
        results = []
        for title, reviews in paper_reviews.items():
            # 提取评审文本
            review_texts = [r['review_text'] for r in reviews]
            
            # 使用统一方法处理单条或多条review
            processed_review = self.aggregate_reviews(review_texts)
            
            results.append({
                'title': title,
                'reviews_count': len(reviews),
                'conference': reviews[0]['conference'],
                'year': reviews[0]['year'],
                'aggregated_review': processed_review,
                'original_ratings': [r['rating'] for r in reviews if r['rating'] is not None],
                'original_confidences': [r['confidence'] for r in reviews if r['confidence'] is not None]
            })
        
        return results
    
    def save_results(self, results, output_dir):
        """保存聚合结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON文件
        with open(os.path.join(output_dir, 'aggregated_reviews.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 为每篇论文创建单独的文件
        for result in results:
            # 使用标题的前30个字符作为文件名（移除特殊字符）
            safe_title = "".join(c for c in result['title'][:30] if c.isalnum() or c.isspace()).strip()
            filename = f"{safe_title}_{result['year']}.txt"
            
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(f"Title: {result['title']}\n")
                f.write(f"Conference: {result['conference']}\n")
                f.write(f"Year: {result['year']}\n")
                f.write(f"Number of Reviews: {result['reviews_count']}\n")
                if result['original_ratings']:
                    f.write(f"Original Ratings: {', '.join(map(str, result['original_ratings']))}\n")
                if result['original_confidences']:
                    f.write(f"Original Confidences: {', '.join(map(str, result['original_confidences']))}\n")
                f.write("\nAggregated Review:\n")
                f.write(result['aggregated_review'])
            
        return len(results)