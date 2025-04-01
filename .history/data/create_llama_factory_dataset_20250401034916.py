import os
import json
import glob
import re
from pathlib import Path

def load_aggregated_reviews(json_path):
    """加载聚合评审数据"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"无法加载聚合评审数据: {e}")
        print("将创建新的聚合评审数据...")
        return []

def generate_aggregated_reviews(extracted_texts_dir, output_json_path):
    """从aggregated_reviews.json加载聚合评审数据"""
    try:
        # 直接加载现有的聚合评审数据
        with open(output_json_path, 'r', encoding='utf-8') as f:
            aggregated_reviews = json.load(f)
        
        print(f"已加载 {len(aggregated_reviews)} 条聚合评审数据")
        return aggregated_reviews
    except Exception as e:
        print(f"加载聚合评审数据失败: {e}")
        return []

def generate_aggregated_reviews(extracted_texts_dir, output_json_path):
    """从提取的文本文件生成聚合评审数据"""
    aggregated_reviews = []
    
    # 递归搜索所有子目录中的.mmd文件
    for mmd_file in glob.glob(os.path.join(extracted_texts_dir, '**', '*.mmd'), recursive=True):
        try:
            # 从文件名中提取ID
            file_name = os.path.basename(mmd_file)
            paper_id = os.path.splitext(file_name)[0]
            
            # 读取.mmd文件内容
            with open(mmd_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取标题
            title_match = re.search(r'^#\s+(.*?)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else "Unknown Title"
            
            # 提取会议信息（如果有）
            conference = ""
            year = ""
            
            # 提取评审内容 - 查找特定的评审部分
            review_match = re.search(r'(?:^|\n)##\s*Review\s*(?:\n|$)(.*?)(?=\n##|\Z)', content, re.DOTALL|re.IGNORECASE)
            review_content = review_match.group(1).strip() if review_match else "No review content found"
            
            # 创建聚合评审数据项
            review_item = {
                "id": paper_id,
                "title": title,
                "conference": conference,
                "year": year,
                "aggregated_review": review_content  # 使用实际提取的评审内容
            }
            
            aggregated_reviews.append(review_item)
            
        except Exception as e:
            print(f"处理文件 {mmd_file} 时出错: {e}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # 保存聚合评审数据
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_reviews, f, ensure_ascii=False, indent=2)
    
    print(f"已生成 {len(aggregated_reviews)} 条聚合评审数据")
    return aggregated_reviews

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
    
    for review in aggregated_reviews:
        paper_id = review['id']
        
        # 跳过已存在的数据
        if paper_id in existing_ids:
            continue
        
        # 查找匹配的PDF解析数据
        if paper_id in parsed_pdfs:
            pdf_data = parsed_pdfs[paper_id]
            
            # 提取论文内容
            paper_content = ""
            if 'abstract' in pdf_data:
                paper_content += f"Abstract: {pdf_data['abstract']}\n\n"
            if 'sections' in pdf_data:
                for section in pdf_data['sections']:
                    if 'heading' in section and 'text' in section:
                        paper_content += f"{section['heading']}\n{section['text']}\n\n"
            
            # 创建数据项
            data_item = {
                "id": paper_id,
                "title": review['title'],
                "conference": review.get('conference', ''),
                "year": review.get('year', ''),
                "paper_content": paper_content,
                "aggregated_review": review['aggregated_review']
            }
            
            new_data.append(data_item)
            matched_count += 1
    
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
    extracted_texts_dir = os.path.join(base_dir, "data", "extracted_texts")
    pdfs_dir = os.path.join(base_dir, "data", "extracted_texts")
    aggregated_reviews_path = os.path.join(base_dir, "data", "aggregated_reviews", "aggregated_reviews.json")
    output_dataset_path = os.path.join(base_dir, "data", "paper_review_dataset", "paper_review_dataset.json")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_dataset_path), exist_ok=True)
    
    # 生成新的聚合评审数据
    print("基于提取的文本生成聚合评审数据...")
    aggregated_reviews = generate_aggregated_reviews(extracted_texts_dir, aggregated_reviews_path)
    
    # 查找解析后的PDF文件
    print("查找解析后的PDF文件...")
    parsed_pdfs = find_parsed_pdfs(pdfs_dir)
    print(f"找到 {len(parsed_pdfs)} 个解析后的PDF文件")
    
    # 创建数据集
    print("创建论文评审数据集...")
    combined_data = create_paper_review_dataset(aggregated_reviews, parsed_pdfs, output_dataset_path)
    
    print("数据集创建完成！可以上传到Hugging Face了")