import openreview
import json
import os
import re
import time
from datetime import datetime

# 创建输出目录
OUTPUT_DIR = "/data/openreview/"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# CONFERENCES = ["ICML.cc", "AAAI.org", "NeurIPS.cc", "aclweb.org/ACL", "ACM.org", "EMNLP", "aclweb.org/NAACL", "COLING.org", "ICLR.cc", "ACCV", "CVPR"]

username = os.getenv('OPENREVIEW_USERNAME_HH')
password = os.getenv('OPENREVIEW_PASSWORD_HH')

# unique_review_types = set()


# def save_review_types(filename="review_types.txt"):
#     """Save collected unique review types to a file"""
#     with open(filename, "w", encoding="utf-8") as file:
#         for review_type in sorted(unique_review_types):  # Sort for readability
#             file.write(review_type + "\n")

#     print(f"✅ Saved {len(unique_review_types)} unique review types to {filename}")

def save_to_txt(subgroups, conference_dir):
    """将子群组列表保存到会议目录下的 TXT 文件"""
    filename = os.path.join(conference_dir, "subgroups.txt")
    with open(filename, "w", encoding="utf-8") as file:
        for group_id in subgroups:
            file.write(group_id + "\n")
    print(f"✅ 成功保存 {len(subgroups)} 个子群组到 {filename}")

def save_to_json(subgroups, filename="subgroups.json"):
    """将子群组列表保存到 JSON 文件"""
    with open(filename, "w") as file:
        json.dump({"subgroups": subgroups}, file, indent=4)
    print(f"✅ 成功保存 {len(subgroups)} 个子群组到 {filename}")

def save_summary_to_txt(conference, total_papers, total_reviews, conference_dir):
    """将会议统计信息保存到会议目录下的 summary.txt"""
    filename = os.path.join(conference_dir, "summary.txt")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"Conference: {conference}\n")
        file.write(f"Total Papers: {total_papers}\n")
        file.write(f"Total Reviews: {total_reviews}\n")
        file.write("-" * 40 + "\n")

def save_results(results, conference_dir):
    """保存测试结果到 JSON 文件"""
    filename = os.path.join(conference_dir, "results.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


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


# def load_subgroups(json_file="subgroups.json"):
#     """从 JSON 文件加载所有 venue_id"""
#     with open(json_file, "r", encoding="utf-8") as f:
#         data = json.load(f)
#         return data.get("subgroups", [])


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
    try:
        venue_group = client.get_group(venue_id)

        if not hasattr(venue_group, "content") or "submission_name" not in venue_group.content:
            return None

        submission_name = venue_group.content["submission_name"]["value"]
        invitation_format = f"{venue_id}/-/{submission_name}"
        submissions = client.get_all_notes(invitation=invitation_format, details="replies")

        if not submissions:
            return None

        processed_papers = []
        total_reviews = 0
        conference_name = venue_id.split('.')[0]

        for paper in submissions:
            
            # we only want new papers starting from 2020
            year = extract_year(paper)
            if year and year < 2020:
                continue

            # 基础论文信息
            paper_info = {
                "title": paper.content.get("title", {}).get("value", "") if isinstance(paper.content.get("title"), dict) else paper.content.get("title", ""),
                "conference": conference_name,
                "pdf_url": f"https://openreview.net/pdf?id={paper.id}",
                "year": year,
                "reviews": []
            }
           
            allowed_review_types = ["Official_Review", "Official_Comment", "Meta_Review"]
            if hasattr(paper, "details") and "replies" in paper.details:
                reviews_count = 0
                for reply in paper.details["replies"]:
                    if "invitations" in reply and reply["invitations"]:
                        invitation_type = reply["invitations"][0].split("/")[-1]
                        
                        if invitation_type in allowed_review_types:
                            review_data = {
                                "type": invitation_type,
                                "content": reply.get("content", {}),
                                "ratings": {}
                            }
                            
                            if "rating" in reply.get("content", {}):
                                review_data["ratings"]["rating"] = reply["content"]["rating"]
                            if "confidence" in reply.get("content", {}):
                                review_data["ratings"]["confidence"] = reply["content"]["confidence"]
                            
                            paper_info["reviews"].append(review_data)
                            reviews_count += 1
                
                total_reviews += reviews_count

            if total_reviews > 0:   
                processed_papers.append(paper_info)

        # 只有当有处理过的论文时才返回结果
        if processed_papers:
            return {
                "venue_id": venue_id,
                "papers": processed_papers,
                "total_papers": len(processed_papers),
                "total_reviews": total_reviews
            }
        return None

    except Exception as e:
        # print(f"处理 venue {venue_id} 时出错: {str(e)}")
        return None

def main():
    """主流程：读取所有 subgroups，获取论文和评审信息"""
    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password
    )

    results = []
    all_subgroups = []
    base_dir = os.path.join(os.path.dirname(__file__), "openreview")

    # CONFERENCES = openreview.tools.get_all_venues(client)
    # CONFERENCES = ['ICML.cc']
    CONFERENCES = ["ICML.cc", "AAAI.org", "ACM.org", "NeurIPS.cc", "EMNLP", "CVPR"]

    for conference in CONFERENCES:
        
        print(f"正在处理会议: {conference}")

        # 设置会议数据文件路径
        conf_name = conference.split('.')[0]  # 'ICML.cc' -> 'ICML'
        conference_dir = os.path.join(base_dir, conf_name)

        # 检查会议文件夹是否已存在，如果存在则跳过，否则创建并爬取数据
        if os.path.exists(conference_dir):
            print(f"⏩ 跳过已存在的会议文件夹: {conf_name}")
            continue
        os.makedirs(conference_dir, exist_ok=True)

        subgroups = get_all_subgroups(client, conference)
        all_subgroups.extend(subgroups)

        total_papers = 0
        total_reviews = 0
        conference_results = []

        for venue_id in subgroups:
            
            result = process_venue(client, venue_id)
            if result:
                conference_results.append(result)
                total_papers += result["total_papers"]
                total_reviews += result["total_reviews"]

        print(f"✅ 完成处理会议: {conference}")
        # print(conference_results)
        
        save_summary_to_txt(conference, total_papers, total_reviews, conference_dir)
        save_results(conference_results, conference_dir)
        

    # 保存子群组到文件
    #save_to_txt(all_subgroups, conference_dir)
    # save_to_json(all_subgroups, conference_dir)
    #save_results(results, conference_dir)
    # save_review_types(os.path.join(OUTPUT_DIR, "review_types.txt"))

if __name__ == "__main__":
    main()