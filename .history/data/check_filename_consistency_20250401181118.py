import os
import json
import re
from pathlib import Path

def check_filename_consistency():
    """检查extracted_texts目录下的.mmd文件与aggregated_reviews目录下文件的命名一致性"""
    base_dir = Path("/Users/arist/Documents/llm-review-sys")
    extracted_dir = base_dir / "data" / "extracted_texts"
    aggregated_dir = base_dir / "data" / "aggregated_reviews"
    
    # 获取所有.mmd文件
    mmd_files = list(extracted_dir.glob("**/*.mmd"))
    print(f"找到 {len(mmd_files)} 个.mmd文件")
    
    # 获取所有.txt文件和aggregated_reviews.json
    txt_files = list(aggregated_dir.glob("*.txt"))
    print(f"找到 {len(txt_files)} 个.txt文件")
    
    # 检查aggregated_reviews.json是否存在
    json_path = aggregated_dir / "aggregated_reviews.json"
    json_data = []
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print(f"aggregated_reviews.json包含 {len(json_data)} 条记录")
    else:
        print("警告: aggregated_reviews.json不存在")
    
    # 提取文件名模式
    mmd_patterns = {}
    for mmd_file in mmd_files:
        filename = mmd_file.stem
        # 检查是否包含下划线（可能是id_year格式）
        if '_' in filename:
            parts = filename.split('_')
            paper_id = parts[0]
            pattern = "id_suffix"
        else:
            paper_id = filename
            pattern = "id"
        
        mmd_patterns[paper_id] = {
            "filename": filename,
            "pattern": pattern,
            "path": str(mmd_file)
        }
    
    # 提取txt文件名模式
    txt_patterns = {}
    for txt_file in txt_files:
        filename = txt_file.stem
        # 检查是否包含下划线（可能是id_year格式）
        if '_' in filename:
            parts = filename.split('_')
            paper_id = parts[0]
            pattern = "id_suffix"
        else:
            paper_id = filename
            pattern = "id"
        
        txt_patterns[paper_id] = {
            "filename": filename,
            "pattern": pattern,
            "path": str(txt_file)
        }
    
    # 提取JSON数据中的ID格式
    json_patterns = {}
    for item in json_data:
        paper_id = item.get('id', '')
        if not paper_id:
            continue
            
        # 检查文件名格式
        filename = f"{paper_id}"
        if 'year' in item:
            year = item.get('year', '')
            if year:
                filename_with_year = f"{paper_id}_{year}"
                json_patterns[paper_id] = {
                    "filename": filename,
                    "filename_with_year": filename_with_year,
                    "has_year": True
                }
            else:
                json_patterns[paper_id] = {
                    "filename": filename,
                    "has_year": False
                }
        else:
            json_patterns[paper_id] = {
                "filename": filename,
                "has_year": False
            }
    
    # 分析不匹配情况
    print("\n=== 分析结果 ===")
    
    # 1. 检查.mmd文件与.txt文件的ID匹配情况
    mmd_ids = set(mmd_patterns.keys())
    txt_ids = set(txt_patterns.keys())
    json_ids = set(json_patterns.keys())
    
    # 在.mmd中存在但在.txt中不存在的ID
    mmd_only = mmd_ids - txt_ids
    if mmd_only:
        print(f"\n在extracted_texts中存在但在aggregated_reviews中不存在的ID: {len(mmd_only)}")
        for i, paper_id in enumerate(sorted(list(mmd_only)[:10])):
            print(f"  {i+1}. {paper_id} -> {mmd_patterns[paper_id]['filename']}.mmd")
        if len(mmd_only) > 10:
            print(f"  ... 以及其他 {len(mmd_only) - 10} 个")
    
    # 在.txt中存在但在.mmd中不存在的ID
    txt_only = txt_ids - mmd_ids
    if txt_only:
        print(f"\n在aggregated_reviews中存在但在extracted_texts中不存在的ID: {len(txt_only)}")
        for i, paper_id in enumerate(sorted(list(txt_only)[:10])):
            print(f"  {i+1}. {paper_id} -> {txt_patterns[paper_id]['filename']}.txt")
        if len(txt_only) > 10:
            print(f"  ... 以及其他 {len(txt_only) - 10} 个")
    
    # 2. 检查命名模式不一致的情况
    inconsistent_patterns = []
    for paper_id in mmd_ids.intersection(txt_ids):
        mmd_pattern = mmd_patterns[paper_id]["pattern"]
        txt_pattern = txt_patterns[paper_id]["pattern"]
        
        if mmd_pattern != txt_pattern:
            inconsistent_patterns.append({
                "paper_id": paper_id,
                "mmd_filename": mmd_patterns[paper_id]["filename"],
                "txt_filename": txt_patterns[paper_id]["filename"],
                "mmd_pattern": mmd_pattern,
                "txt_pattern": txt_pattern
            })
    
    if inconsistent_patterns:
        print(f"\n命名模式不一致的文件: {len(inconsistent_patterns)}")
        for i, item in enumerate(inconsistent_patterns[:10]):
            print(f"  {i+1}. {item['paper_id']}: {item['mmd_filename']}.mmd vs {item['txt_filename']}.txt")
        if len(inconsistent_patterns) > 10:
            print(f"  ... 以及其他 {len(inconsistent_patterns) - 10} 个")
    
    # 3. 检查JSON数据中的ID与文件名的匹配情况
    json_mmd_mismatch = []
    for paper_id in json_ids.intersection(mmd_ids):
        json_item = json_patterns[paper_id]
        mmd_filename = mmd_patterns[paper_id]["filename"]
        
        # 检查是否使用了带年份的文件名
        if json_item.get("has_year", False):
            expected_filename = json_item["filename_with_year"]
            if mmd_filename != expected_filename and mmd_filename != paper_id:
                json_mmd_mismatch.append({
                    "paper_id": paper_id,
                    "mmd_filename": mmd_filename,
                    "expected_filename": expected_filename,
                    "or_expected": paper_id
                })
    
    if json_mmd_mismatch:
        print(f"\nJSON数据与.mmd文件名不匹配: {len(json_mmd_mismatch)}")
        for i, item in enumerate(json_mmd_mismatch[:10]):
            print(f"  {i+1}. {item['paper_id']}: 当前={item['mmd_filename']}.mmd, 期望={item['expected_filename']}.mmd 或 {item['or_expected']}.mmd")
        if len(json_mmd_mismatch) > 10:
            print(f"  ... 以及其他 {len(json_mmd_mismatch) - 10} 个")
    
    # 4. 检查JSON数据中的ID与.txt文件名的匹配情况
    json_txt_mismatch = []
    for paper_id in json_ids.intersection(txt_ids):
        json_item = json_patterns[paper_id]
        txt_filename = txt_patterns[paper_id]["filename"]
        
        # 检查是否使用了带年份的文件名
        if json_item.get("has_year", False):
            expected_filename = json_item["filename_with_year"]
            if txt_filename != expected_filename and txt_filename != paper_id:
                json_txt_mismatch.append({
                    "paper_id": paper_id,
                    "txt_filename": txt_filename,
                    "expected_filename": expected_filename,
                    "or_expected": paper_id
                })
    
    if json_txt_mismatch:
        print(f"\nJSON数据与.txt文件名不匹配: {len(json_txt_mismatch)}")
        for i, item in enumerate(json_txt_mismatch[:10]):
            print(f"  {i+1}. {item['paper_id']}: 当前={item['txt_filename']}.txt, 期望={item['expected_filename']}.txt 或 {item['or_expected']}.txt")
        if len(json_txt_mismatch) > 10:
            print(f"  ... 以及其他 {len(json_txt_mismatch) - 10} 个")
    
    # 总结
    print("\n=== 总结 ===")
    print(f"总共检查了 {len(mmd_files)} 个.mmd文件和 {len(txt_files)} 个.txt文件")
    print(f"JSON数据中包含 {len(json_ids)} 个唯一ID")
    
    if not mmd_only and not txt_only and not inconsistent_patterns and not json_mmd_mismatch and not json_txt_mismatch:
        print("恭喜！所有文件命名都是一致的。")
    else:
        print("发现命名不一致的情况，建议进行修正。")
        
        # 提供修正建议
        print("\n=== 修正建议 ===")
        print("1. 统一使用纯ID作为文件名 (推荐)")
        print("   - 将所有文件重命名为 {paper_id}.mmd 和 {paper_id}.txt")
        print("2. 统一使用ID_YEAR格式")
        print("   - 将所有文件重命名为 {paper_id}_{year}.mmd 和 {paper_id}_{year}.txt")
        print("3. 确保.mmd文件和对应的.txt文件使用相同的命名模式")

if __name__ == "__main__":
    check_filename_consistency()