from openai import OpenAI
import json
import pandas as pd
import os

class ReviewAggregator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def aggregate_reviews(self, reviews):
        """使用GPT API聚合多条review"""
        prompt = f"""
        Please aggregate the following reviews into a unified format:
        1. Summary
        2. Strengths
        3. Weaknesses
        4. Suggestions for Improvement
        5. Review Result
        
        Reviews:
        {json.dumps(reviews, ensure_ascii=False)}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def process_openreview_dataset(self, dataset_path):
        """处理OpenReview格式的数据集并聚合评审意见"""
        # 读取OpenReview数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 按论文ID组织评审
        paper_reviews = {}
        
        # 处理OpenReview格式的数据
        for item in data:
            # 假设每个item包含forum(论文ID)和content(评审内容)
            paper_id = item.get('forum')
            
            # 检查是否为评审类型的记录
            if 'content' in item and paper_id:
                # 提取评审内容
                review_content = {}
                
                # 常见的OpenReview评审字段
                review_fields = ['summary', 'strengths', 'weaknesses', 'comments', 'rating', 'confidence']
                
                for field in review_fields:
                    if field in item['content']:
                        review_content[field] = item['content'][field]
                
                # 如果提取到了评审内容
                if review_content:
                    if paper_id not in paper_reviews:
                        paper_reviews[paper_id] = []
                    
                    # 将评审内容格式化为字符串
                    review_text = ""
                    for field, content in review_content.items():
                        review_text += f"{field.capitalize()}: {content}\n\n"
                    
                    paper_reviews[paper_id].append(review_text)
        
        # 聚合每篇论文的评审
        results = []
        for paper_id, reviews in paper_reviews.items():
            # 如果只有一条评审，则跳过聚合
            if len(reviews) <= 1:
                continue
                
            # 聚合评审意见
            aggregated_review = self.aggregate_reviews(reviews)
            
            results.append({
                'paper_id': paper_id,
                'reviews_count': len(reviews),
                'aggregated_review': aggregated_review
            })
        
        return results
    
    def process_dataset(self, dataset_path):
        """从数据集中找出同一论文的不同评审意见并聚合"""
        # 如果是OpenReview格式的JSON文件，使用专门的处理方法
        if dataset_path.endswith('openreview_dataset.json'):
            return self.process_openreview_dataset(dataset_path)
            
        # 读取数据集
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
            df = pd.read_excel(dataset_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV, JSON, or Excel file.")
        
        # 确保数据集包含必要的列
        required_columns = ['paper_id', 'review']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain the following columns: {required_columns}")
        
        # 按论文ID分组
        results = []
        for paper_id, group in df.groupby('paper_id'):
            reviews = group['review'].tolist()
            
            # 如果只有一条评审，则跳过聚合
            if len(reviews) <= 1:
                continue
                
            # 聚合评审意见
            aggregated_review = self.aggregate_reviews(reviews)
            
            results.append({
                'paper_id': paper_id,
                'reviews_count': len(reviews),
                'aggregated_review': aggregated_review
            })
        
        return results
    
    def save_results(self, results, output_dir):
        """保存聚合结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON文件
        with open(os.path.join(output_dir, 'aggregated_reviews.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 为每个论文创建单独的文件
        for result in results:
            paper_id = result['paper_id']
            with open(os.path.join(output_dir, f'paper_{paper_id}_review.txt'), 'w', encoding='utf-8') as f:
                f.write(result['aggregated_review'])
        
        return len(results)