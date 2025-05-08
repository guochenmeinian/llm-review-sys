import os
import json
import glob
import re
from pathlib import Path

def load_aggregated_reviews(json_path):
    """加载聚合评审数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_parsed_pdfs(base_dir):
    """查找所有解析后的PDF文件"""
    parsed_files = {}
    
    # 递归搜索所有子目录中的.mmd文件
    for mmd_file in glob.glob(os.path.join(base_dir, '**', '*.mmd'), recursive=True):
        try:
            # 从文件名中提取ID
            file_name = os.path.basename(mmd_file)
            paper_id = os.path.splitext(file_name)[0]
            
            # 读取.mmd文件内容
            with open(mmd_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析.mmd文件内容，提取标题、摘要和正文
            title = ""
            abstract = ""
            sections = []
            
            # 简单解析.mmd文件结构
            current_section = None
            current_text = []
            
            for line in content.split('\n'):
                if line.startswith('# '):  # 标题
                    title = line[2:].strip()
                elif line.startswith('## Abstract'):  # 摘要部分
                    if current_section and current_text:
                        sections.append({
                            'heading': current_section,
                            'text': '\n'.join(current_text)
                        })
                    current_section = "Abstract"
                    current_text = []
                elif line.startswith('## '):  # 其他章节
                    # 保存之前的章节
                    if current_section:
                        if current_section == "Abstract":
                            abstract = '\n'.join(current_text)
                        else:
                            sections.append({
                                'heading': current_section,
                                'text': '\n'.join(current_text)
                            })
                    
                    current_section = line[3:].strip()
                    current_text = []
                else:
                    current_text.append(line)
            
            # 保存最后一个章节
            if current_section and current_text:
                if current_section == "Abstract":
                    abstract = '\n'.join(current_text)
                else:
                    sections.append({
                        'heading': current_section,
                        'text': '\n'.join(current_text)
                    })
            
            # 构建解析后的数据结构
            parsed_files[paper_id] = {
                'id': paper_id,
                'title': title,
                'abstract': abstract,
                'sections': sections,
                'file_path': mmd_file
            }
            
        except Exception as e:
            print(f"无法解析文件 {mmd_file}: {e}")
    
    return parsed_files

def create_paper_review_dataset(aggregated_reviews, parsed_pdfs, output_path):
    """创建论文评审数据集"""
    # 检查输出目录是否存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 如果输出文件已存在，加载现有数据
    existing_data = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"已加载现有数据集，包含 {len(existing_data)} 条记录")
        except Exception as e:
            print(f"加载现有数据集失败: {e}")
            existing_data = []
    
    # 创建ID到现有数据的映射，用于检查重复
    existing_ids = set()
    for item in existing_data:
        if 'id' in item:
            existing_ids.add(item['id'])
    
    # 合并数据
    new_data = []
    matched_count = 0
    
    # 添加不匹配记录统计
    unmatched_ids = []
    matched_ids = []
    
    for review in aggregated_reviews:
        paper_id = review['id']
        
        if paper_id in existing_ids:
            continue
            
        # 获取原始评分和置信度
        original_ratings = review.get('original_ratings', [])
        original_confidences = review.get('original_confidences', [])
        
        # 如果没有原始评分或置信度，直接跳过
        if not original_ratings or not original_confidences:
            unmatched_ids.append(paper_id)
            continue
            
        if paper_id in parsed_pdfs:
            pdf_data = parsed_pdfs[paper_id]
            
            # 提取论文内容
            paper_content = ""
            if 'abstract' in pdf_data:
                paper_content += f"{pdf_data['abstract']}\n\n"
            if 'sections' in pdf_data:
                for section in pdf_data['sections']:
                    if 'heading' in section and 'text' in section:
                        paper_content += f"{section['heading']}\n{section['text']}\n\n"
            
            review_content = review.get('review') or review.get('aggregated_review', '')
            
            # 计算平均评分和平均置信度
            valid_ratings = [r for r in original_ratings if r != -1]
            valid_confidences = [c for c in original_confidences if c != -1]
            
            # 只有当有有效评分和置信度时才添加数据
            if valid_ratings and valid_confidences:
                avg_rating = sum(valid_ratings) / len(valid_ratings)
                avg_confidence = sum(valid_confidences) / len(valid_confidences)
                
                data_item = {
                    "id": paper_id,
                    "title": review['title'],
                    "conference": review.get('conference', ''),
                    "year": review.get('year', ''),
                    "paper_content": paper_content,
                    "aggregated_review": review_content,
                    "original_ratings": original_ratings,
                    "original_confidences": original_confidences,
                    "avg_rating": round(avg_rating, 2),
                    "avg_confidence": round(avg_confidence, 2)
                }
                
                new_data.append(data_item)
                matched_count += 1
                matched_ids.append(paper_id)
            else:
                unmatched_ids.append(paper_id)
        else:
            unmatched_ids.append(paper_id)
    
    # 添加调试信息
    print(f"\n匹配统计:")
    print(f"总评审数: {len(aggregated_reviews)}")
    print(f"已存在记录: {len(existing_ids)}")
    print(f"成功匹配: {len(matched_ids)}")
    print(f"未匹配: {len(unmatched_ids)}")
    
    if len(unmatched_ids) > 0:
        print("\n前10个未匹配的paper_id:")
        print(unmatched_ids[:10])
    
    # 合并现有数据和新数据
    combined_data = existing_data + new_data
    
    # 保存合并后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据集创建完成！")
    print(f"匹配并添加了 {matched_count} 条新记录")
    print(f"数据集现在包含 {len(combined_data)} 条记录")
    
    return combined_data

if __name__ == "__main__":
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdfs_dir = os.path.join(base_dir, "data", "extracted_texts")
    aggregated_reviews_path = os.path.join(base_dir, "data", "aggregated_reviews", "aggregated_reviews.json")
    output_dataset_path = os.path.join(base_dir, "data", "draft_dataset", "draft_dataset.json")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_dataset_path), exist_ok=True)
    
    # 加载聚合评审数据
    print("加载聚合评审数据...")
    aggregated_reviews = load_aggregated_reviews(aggregated_reviews_path)
    print(f"已加载 {len(aggregated_reviews)} 条聚合评审")
    
    # 查找解析后的PDF文件
    print("查找解析后的PDF文件...")
    parsed_pdfs = find_parsed_pdfs(pdfs_dir)
    print(f"找到 {len(parsed_pdfs)} 个解析后的PDF文件")
    
    # 创建数据集
    print("创建论文评审数据集...")
    combined_data = create_paper_review_dataset(aggregated_reviews, parsed_pdfs, output_dataset_path)
    
    print("数据集创建完成！可以上传到Hugging Face了")