import os
import json
import requests
import openreview
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# 加载环境变量（从 .env 文件）
load_dotenv()

# 读取 OpenReview 认证信息
OPENREVIEW_USERNAME = os.getenv("OPENREVIEW_USERNAME")
OPENREVIEW_PASSWORD = os.getenv("OPENREVIEW_PASSWORD")

# 配置 OpenReview API 客户端
client = openreview.Client(
    baseurl='https://api.openreview.net',
    username=OPENREVIEW_USERNAME,
    password=OPENREVIEW_PASSWORD
)

# 目标会议（修改这里以爬取不同会议）
conference_id = 'ICML.cc/2024/Conference#tab-accept-oral'  
batch_size = 500   # 每次爬取多少篇论文
max_papers = 10000  # 总共爬取的论文数
output_dir = "data/json_papers"
os.makedirs(output_dir, exist_ok=True)

# 读取已爬取的 paper_id 记录
existing_papers = set()
for file in os.listdir(output_dir):
    if file.endswith(".json"):
        with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            existing_papers.update([p["paper_id"] for p in data])

print(f"已有 {len(existing_papers)} 篇论文，跳过已爬取的论文")

# 开始爬取论文
paper_data = []
offset = 0
while len(paper_data) + len(existing_papers) < max_papers:
    try:
        # 获取论文
        notes = client.get_notes(invitation=f'{conference_id}/-/Blind_Submission', offset=offset, limit=batch_size)
        if not notes:
            print("没有更多论文了")
            break

        for paper in notes:
            paper_id = paper.id
            if paper_id in existing_papers:
                continue  # 跳过已爬取的论文
            
            title = paper.content.get("title", "N/A")
            abstract = paper.content.get("abstract", "N/A")
            authors = paper.content.get("authors", [])
            pdf_url = f"https://openreview.net{paper.content['pdf']}" if 'pdf' in paper.content else None
            
            # 获取论文的所有评论
            comments = client.get_notes(forum=paper_id)
            reviews = []
            for comment in comments:
                if comment.invitation.endswith('Official_Review'):
                    reviews.append({
                        "review_id": comment.id,
                        "rating": comment.content.get("rating", "N/A"),
                        "confidence": comment.content.get("confidence", "N/A"),
                        "review_text": comment.content.get("review", "N/A")
                    })

            paper_data.append({
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "pdf_url": pdf_url,
                "reviews": reviews
            })

        print(f"已爬取 {len(paper_data)} 篇论文")

        # 每 1000 篇论文存储为一个 JSON 文件
        if len(paper_data) >= 1000:
            file_index = len(os.listdir(output_dir)) + 1
            file_path = os.path.join(output_dir, f"papers_metadata_part{file_index}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(paper_data, f, indent=4, ensure_ascii=False)
            print(f"数据已存储到 {file_path}")
            paper_data = []  # 清空缓存，准备存下一个批次

        offset += batch_size  # 移动到下一个批次

    except Exception as e:
        print(f"发生错误: {e}")
        break  # 发生错误时终止，避免重复爬取

# 处理剩余数据
if paper_data:
    file_index = len(os.listdir(output_dir)) + 1
    file_path = os.path.join(output_dir, f"papers_metadata_part{file_index}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(paper_data, f, indent=4, ensure_ascii=False)
    print(f"数据已存储到 {file_path}")

print("爬取完成！")
