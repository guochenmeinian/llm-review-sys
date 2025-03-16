import os
import re
import argparse
import logging

def remove_references(content):
    """删除参考文献及其之后的内容"""
    # 匹配不同格式的参考文献标题
    ref_pattern = re.compile(r'^#+\s*References?\s*$', flags=re.IGNORECASE | re.MULTILINE)
    match = ref_pattern.search(content)
    return content[:match.start()] if match else content

def process_mmd_files(root_dir, target_conferences):
    """处理指定会议的MMD文件"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建清理目录（保持原目录结构）
    cleaned_root = os.path.join(os.path.dirname(root_dir), "parsed_texts_cleaned")
    os.makedirs(cleaned_root, exist_ok=True)

    for conference in target_conferences:
        # 原会议目录
        src_dir = os.path.join(root_dir, conference)
        # 新清理目录
        dest_dir = os.path.join(cleaned_root, conference)
        
        if not os.path.isdir(src_dir):
            logging.warning(f"会议目录不存在: {src_dir}")
            continue
        
        # 创建目标目录
        os.makedirs(dest_dir, exist_ok=True)

        for mmd_file in os.listdir(src_dir):
            if mmd_file.endswith('.mmd'):
                # 保持目录结构：parsed_texts_cleaned/会议名称/文件名
                dest_path = os.path.join(cleaned_root, conference, mmd_file)
                src_path = os.path.join(src_dir, mmd_file)
                dest_path = os.path.join(dest_dir, mmd_file)
                
                try:
                    # 读取原文件
                    with open(src_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 处理内容
                    new_content = remove_references(content)
                    
                    # 写入新目录
                    with open(dest_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                        
                    logging.info(f"已保存到: {dest_path}")
                except Exception as e:
                    logging.error(f"处理失败: {src_path} - {str(e)}")

def main():
    # 设置目录（与pdf_downloader.py保持一致）
    base_dir = os.path.join(os.path.dirname(__file__))
    parsed_root = os.path.join(base_dir, "parsed_texts")
    
    # 指定目标会议列表（示例）
    TARGET_CONFERENCES = ['AAAI']  # 在此处修改需要处理的会议
    
    # 执行清理操作
    # 修改根目录为原 parsed_texts 目录
    process_mmd_files(parsed_root, TARGET_CONFERENCES)

if __name__ == "__main__":
    main()