import json
from datasets import Dataset

def prepare_dataset(input_dir):
    """准备训练数据集"""
    data = []
    for conference in os.listdir(input_dir):
        results_path = os.path.join(input_dir, conference, "results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                conference_data = json.load(f)
                data.extend(process_conference_data(conference_data))
    
    return Dataset.from_list(data)

def process_conference_data(conference_data):
    """处理单个会议的数据"""
    processed = []
    for venue in conference_data:
        for paper in venue["papers"]:
            if "reviews" in paper:
                processed.append({
                    "title": paper["title"],
                    "reviews": paper["reviews"],
                    "conference": paper["conference"]
                })
    return processed