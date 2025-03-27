from openai import OpenAI
import json
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class ReviewAggregator:
    def __init__(self, api_key_name):
        # 从环境变量获取API密钥
        api_key = os.getenv(api_key_name)
        if not api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)
    
    def aggregate_reviews(self, reviews):
        """使用统一英文prompt处理单条或多条review，处理长评论不截断"""
    
        if not reviews:
            print("No reviews to process.")
            return ""
    
        # 估计token数量（粗略估计：每4个字符约1个token）
        total_chars = sum(len(review) for review in reviews)
        estimated_tokens = total_chars // 4
        
        # 如果评论太长，分批处理
        if estimated_tokens > 6000:  # 为prompt和其他内容预留空间
            print(f"评论内容过长 (估计 {estimated_tokens} tokens)，将分批处理")
            
            # 将评论分成多批次
            batches = []
            current_batch = []
            current_chars = 0
            
            for review in reviews:
                review_chars = len(review)
                
                # 如果当前批次加上这条评论会超过限制，创建新批次
                if current_chars + review_chars > 24000:  # ~6000 tokens
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [review]
                    current_chars = review_chars
                else:
                    current_batch.append(review)
                    current_chars += review_chars
            
            if current_batch:
                batches.append(current_batch)
            
            print(f"将评论分成 {len(batches)} 批处理")
            
            # 处理每个批次并合并结果
            summaries = []
            for i, batch in enumerate(batches):
                print(f"处理第 {i+1}/{len(batches)} 批评论...")
                batch_summary = self._process_single_batch(batch)
                summaries.append(batch_summary)
            
            # 合并所有批次的摘要
            if len(summaries) > 1:
                print("合并所有批次的摘要...")
                final_summary = self._merge_summaries(summaries)
                return final_summary
            else:
                return summaries[0]
        else:
            # 评论不长，直接处理
            return self._process_single_batch(reviews)
    
    def _process_single_batch(self, reviews):
        """处理单批次评论"""
        numbered_reviews = "\n\n".join([f"Review {i+1}: {review}" for i, review in enumerate(reviews)])
    
        prompt = f"""
        Please generate a concise summary based on the following review(s) with the following three sections:
        1. Key points
        2. Strengths and weaknesses
        3. Suggestions for improvement
        
        Critical Requirements:
        - Strictly maintain the exact core viewpoints of the original review(s)
        - Absolute prohibition on adding any new information
        - Summarize ONLY using the provided review content
        - Preserve verbatim any specific technical suggestions or terms
        - Ensure the summary captures the full depth and nuance of the original reviews
        - Match the original review's academic tone and technical precision
        - Reduce length by 20-30% without losing critical information
        - Use FIRST PERSON PLURAL perspective (we, our, us) when referring to reviewers. For example: "We suggest improving the method" instead of "The reviewers suggest improving the method"
        
        Review(s):
        {numbered_reviews}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
    
            if not response.choices:
                print("API call failed or returned no choices.")
                return ""
    
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            return f"处理评论时出错: {str(e)}"
    
    def _merge_summaries(self, summaries):
        """合并多个批次的摘要"""
        combined_summaries = "\n\n".join([f"Summary {i+1}:\n{summary}" for i, summary in enumerate(summaries)])
        
        prompt = f"""
        Please merge the following summaries into a single comprehensive summary with the following three sections:
        1. Key points
        2. Strengths and weaknesses
        3. Suggestions for improvement
        
        Strict Merging Requirements:
        - Preserve the exact core viewpoints from ALL summaries
        - Completely eliminate redundant information
        - Maintain 100% fidelity to the original review contents
        - Ensure the final summary is logically coherent
        - Keep the original academic and technical precision
        - Do NOT introduce any new interpretations or insights
        - Use FIRST PERSON PLURAL perspective (we, our, us) when referring to reviewers. For example: "We suggest improving the method" instead of "The reviewers suggest improving the method"
        
        Summaries to merge:
        {combined_summaries}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
    
            if not response.choices:
                print("API call failed or returned no choices.")
                return ""
    
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"合并摘要时出错: {str(e)}"
            print(error_msg)
            # 记录错误到文件
            self._log_error(error_msg)
            # 如果合并失败，返回所有摘要的简单连接
            return "\n\n===== 摘要分割线 =====\n\n".join(summaries)
        
    def process_openreview_dataset(self, dataset_path, output_dir):
        """处理数据集并聚合评审意见，处理一篇保存一篇"""
    
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            return []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("Dataset is empty.")
            return []
    
        print(f"Loaded {len(data)} items from the dataset.")
    
        # 按论文标题组织评审
        paper_reviews = {}
        paper_ids = {}  # 存储论文ID
        
        for item in data:
            # 提取标题和评审内容
            title = item.get('title')
            paper_id = item.get('id', f"paper_{len(paper_ids) + 1}")  # 使用ID或生成一个
            reviews_array = item.get('reviews', [])
            
            if title and reviews_array:
                # 如果标题不存在，创建新条目
                if title not in paper_reviews:
                    paper_reviews[title] = []
                    paper_ids[title] = paper_id
                
                # 处理每条评审
                for review in reviews_array:
                    review_text = review.get('review_text')
                    if review_text:

                        # 获取评分和置信度
                        rating = review.get('rating')
                        confidence = review.get('confidence')
                        
                        # 如果评分或置信度包含描述性文本，只提取数字部分
                        if isinstance(rating, str) and ':' in rating:
                            try:
                                rating = int(rating.split(':', 1)[0].strip())
                            except ValueError:
                                rating = -1
                        
                        if isinstance(confidence, str) and ':' in confidence:
                            try:
                                confidence = int(confidence.split(':', 1)[0].strip())
                            except ValueError:
                                confidence = -1

                        # 构建结构化的评审信息
                        review_info = {
                            'review_text': review_text,
                            'rating': rating,
                            'confidence': confidence,
                            'year': item.get('year'),
                            'conference': item.get('conference')
                        }
                        
                        paper_reviews[title].append(review_info)
    
        if not paper_reviews:
            print("No valid papers found in the dataset.")
            return []
    
        print(f"Found {len(paper_reviews)} papers with valid reviews.")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取现有的聚合结果
        existing_results_path = os.path.join(output_dir, 'aggregated_reviews.json')
        if os.path.exists(existing_results_path):
            with open(existing_results_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        else:
            existing_results = []
        
        # 获取已处理的标题集合
        titles_in_existing = {result['title'] for result in existing_results}
        
        # 聚合每篇论文的评审，并立即保存
        processed_count = 0
        for title, reviews in tqdm(paper_reviews.items(), desc="Processing papers"):
            try:
                # 如果已经处理过，跳过
                if title in titles_in_existing:
                    continue
                    
                print(f"Processing paper: {title}")
                # 提取评审文本
                review_texts = [r['review_text'] for r in reviews]
                
                # 使用统一方法处理单条或多条review
                processed_review = self.aggregate_reviews(review_texts)

                # 如果处理出错，记录并跳过
                if processed_review.startswith("处理评论时出错") or processed_review.startswith("合并摘要时出错"):
                    error_msg = f"处理论文 '{title}' 时出错: {processed_review}"
                    self._log_error(error_msg)
                    error_count += 1
                    continue
                    
                # 获取论文ID
                paper_id = paper_ids[title]
                
                # 创建结果对象
                result = {
                    'id': paper_id,
                    'title': title,
                    'reviews_count': len(reviews),
                    'conference': reviews[0]['conference'],
                    'year': reviews[0]['year'],
                    'aggregated_review': processed_review,
                    'original_ratings': [r['rating'] for r in reviews if r['rating'] is not None],
                    'original_confidences': [r['confidence'] for r in reviews if r['confidence'] is not None]
                }
                
                # 添加到现有结果并保存
                existing_results.append(result)
                with open(existing_results_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                
                # 使用ID作为文件名
                filename = f"{paper_id}_{result['year']}.txt"
                
                with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(f"ID: {paper_id}\n")
                    f.write(f"Title: {title}\n")
                    f.write(f"Conference: {result['conference']}\n")
                    f.write(f"Year: {result['year']}\n")
                    f.write(f"Number of Reviews: {result['reviews_count']}\n")
                    if result['original_ratings']:
                        f.write(f"Original Ratings: {', '.join(map(str, result['original_ratings']))}\n")
                    if result['original_confidences']:
                        f.write(f"Original Confidences: {', '.join(map(str, result['original_confidences']))}\n")
                    f.write("\nAggregated Review:\n")
                    f.write(result['aggregated_review'])
                
                processed_count += 1
            
            except Exception as e:
                error_msg = f"处理论文 '{title}' 时发生异常: {str(e)}"
                print(error_msg)
                self._log_error(error_msg)
                error_count += 1
        
        print(f"处理完成，共处理了 {processed_count} 篇新论文，{error_count} 篇处理失败。")
        return processed_count


def main():
    # 设置数据集路径和输出目录
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "draft_data", "paper_reviews.json")
    output_dir = os.path.join(base_dir, "aggregated_reviews")

    # 创建ReviewAggregator实例
    aggregator = ReviewAggregator('OPENAI_KEY_CG')

    # 处理数据集并保存结果，一篇一篇处理
    processed_count = aggregator.process_openreview_dataset(dataset_path, output_dir)
    print(f"处理完成，共处理了 {processed_count} 篇新论文。")

if __name__ == "__main__":
    main()