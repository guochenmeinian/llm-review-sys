import openreview
import json
import os
import re
import time
from datetime import datetime

# 创建输出目录
OUTPUT_DIR = "openreview_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CONFERENCES = ["ICML.cc", "AAAI.org", "NeurIPS.cc", "aclweb.org/ACL", "ACM.org", "EMNLP", "aclweb.org/NAACL", "COLING.org", "ICLR.cc", "ACCV", "CVPR"]

username = os.getenv('OPENREVIEW_USERNAME')
password = os.getenv('OPENREVIEW_PASSWORD')

unique_review_types = set()


def save_review_types(filename="review_types.txt"):
    """Save collected unique review types to a file"""
    with open(filename, "w", encoding="utf-8") as file:
        for review_type in sorted(unique_review_types):  # Sort for readability
            file.write(review_type + "\n")

    print(f"✅ Saved {len(unique_review_types)} unique review types to {filename}")

def save_to_txt(subgroups, filename="subgroups.txt"):
    """将子群组列表保存到 TXT 文件"""
    with open(filename, "w") as file:
        for group_id in subgroups:
            file.write(group_id + "\n")
    print(f"✅ 成功保存 {len(subgroups)} 个子群组到 {filename}")

def save_to_json(subgroups, filename="subgroups.json"):
    """将子群组列表保存到 JSON 文件"""
    with open(filename, "w") as file:
        json.dump({"subgroups": subgroups}, file, indent=4)
    print(f"✅ 成功保存 {len(subgroups)} 个子群组到 {filename}")

def get_all_subgroups(client, parent_group):
    """获取所有子群组"""
    subgroups = set()
    try:
        groups = client.get_groups(id=parent_group)
        for group in groups:
            subgroups.add(group.id)
            temp_subgroups = client.get_groups(id=group.id + ".*")  # 获取该组的所有子群组
            for subgroup in temp_subgroups:
                subgroups.add(subgroup.id)
    except openreview.OpenReviewException:
        pass  # 跳过访问失败的会议
    return list(subgroups)


def save_summary_to_txt(conference, total_papers, total_reviews, total_comments, filename="summary.txt"):
    """将每个 Conference 的总论文数和总评审数追加保存到 TXT 文件"""
    with open(filename, "a", encoding="utf-8") as file:  # 以追加模式写入
        file.write(f"Venue: {conference}\n")
        file.write(f"  Total Papers: {total_papers}\n")
        file.write(f"  Total Reviews: {total_reviews}\n")
        file.write(f"  Total Comments: {total_comments}\n")
        file.write("-" * 40 + "\n")


# def load_subgroups(json_file="subgroups.json"):
#     """从 JSON 文件加载所有 venue_id"""
#     with open(json_file, "r", encoding="utf-8") as f:
#         data = json.load(f)
#         return data.get("subgroups", [])



def save_results(results, filename="test_results.json"):
    """保存测试结果到 JSON 文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def extract_year(paper):
    """从论文 metadata 或创建时间中提取年份"""
    if hasattr(paper, "content") and "year" in paper.content:
        try:
            year = int(paper.content["year"]["value"]) if isinstance(paper.content["year"], dict) else int(paper.content["year"])
            return year
        except (ValueError, TypeError):
            pass  # 如果 year 解析失败，继续尝试 cdate

    # 尝试从 cdate 获取年份（cdate 是论文的创建时间，单位是毫秒）
    if hasattr(paper, "cdate"):
        try:
            import datetime
            year = datetime.datetime.utcfromtimestamp(paper.cdate / 1000).year
            return year
        except (ValueError, TypeError):
            return None

    return None  # 如果都无法获取，则返回 None

def process_venue(client, venue_id):
    """处理单个 venue_id，获取论文和评审信息"""
    try:
        venue_group = client.get_group(venue_id)

        if not hasattr(venue_group, "content") or "submission_name" not in venue_group.content:
            return None  # 跳过不兼容的 venue

        submission_name = venue_group.content["submission_name"]["value"]
        invitation_format = f"{venue_id}/-/{submission_name}"

        submissions = client.get_all_notes(invitation=invitation_format, details="replies")

        if not submissions:
            return None  # 没有找到论文，跳过

        max_papers = min(5, len(submissions))
        processed_papers = []
        total_reviews = 0
        total_comments = 0

        for paper_idx in range(max_papers):
            paper = submissions[paper_idx]
            paper_id = paper.id
            forum_id = paper.forum if hasattr(paper, "forum") else paper.id

            title = "无标题"
            if hasattr(paper, "content") and "title" in paper.content:
                title = paper.content["title"]["value"] if isinstance(paper.content["title"], dict) else paper.content["title"]

            # year = extract_year(paper)
            # print(year)
           
            reviews = []
            comments = []

            if hasattr(paper, "details") and "replies" in paper.details:
                for reply in paper.details["replies"]:
                    if "invitations" in reply and reply["invitations"]:
                        invitation_type = reply["invitations"][0].split("/")[-1]  # Extract only the type name
                        unique_review_types.add(invitation_type)  # Add to set (no duplicates)

                    if "Official_Review" in reply["invitations"][0]:
                        reviews.append(reply)
                    elif "Official_Comment" in reply["invitations"][0] or "Comment" in reply["invitations"][0]:
                        comments.append(reply)

            total_reviews += len(reviews)
            total_comments += len(comments)

            if not reviews:
                continue 
            
            total_reviews += len(reviews)
            total_comments += len(comments)

            # 添加到处理结果
            paper_result = {
                "id": paper_id,
                "title": title,
                # "year": year if year else "未知",
                "paper_data": paper.to_json() if hasattr(paper, "to_json") else {"id": paper_id},
                "reviews": reviews,
                "comments": comments
            }
            processed_papers.append(paper_result)

        return {
            "venue_id": venue_id,
            "total_papers": len(processed_papers),
            "total_reviews": total_reviews,
            "total_comments": total_comments,
            "papers": processed_papers
        } if processed_papers else None  # 如果没有符合条件的论文，则返回 None

    except Exception:
        return None  # 访问失败的直接跳过

def main():
    """主流程：读取所有 subgroups，获取论文和评审信息"""
    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password
    )

    results = []
    all_subgroups = []

    CONFERENCES = openreview.tools.get_all_venues(client)
    # CONFERENCES = ['ICML.cc']

    for conference in CONFERENCES:
        subgroups = get_all_subgroups(client, conference)
        all_subgroups.extend(subgroups)

        total_papers = 0
        total_reviews = 0
        total_comments = 0
        conference_results = []

        for venue_id in subgroups:
            result = process_venue(client, venue_id)

            if result:
                conference_results.append(result)
                results.append(result)
                total_papers += result["total_papers"]
                total_reviews += result["total_reviews"]
                total_comments += result["total_comments"]

        save_summary_to_txt(conference, total_papers, total_reviews, total_comments)
        

    # 保存子群组到文件
    save_to_txt(all_subgroups)
    save_to_json(all_subgroups)
    save_results(results, os.path.join(OUTPUT_DIR, "test_results.json"))
    save_review_types(os.path.join(OUTPUT_DIR, "review_types.txt"))

if __name__ == "__main__":
    main()