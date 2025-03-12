from openai import OpenAI
import json

class ReviewAggregator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def aggregate_reviews(self, reviews):
        """使用GPT API聚合多条review"""
        prompt = f"""
        请将以下评审意见聚合为统一的格式：
        1. 总结
        2. 优点
        3. 缺点
        4. 改进建议
        5. 评审结果
        
        评审意见：
        {json.dumps(reviews, ensure_ascii=False)}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content