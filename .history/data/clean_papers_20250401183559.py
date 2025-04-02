import os
import re
import glob
from tqdm import tqdm

def process_all_papers(input_dir, output_dir):
    """处理指定目录下的所有论文文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有.mmd文件，包括子目录中的文件
    mmd_files = glob.glob(os.path.join(input_dir, '**', '*.mmd'), recursive=True)
    print(f"找到 {len(mmd_files)} 个.mmd文件")
    
    for mmd_file in tqdm(mmd_files, desc="处理论文"):
        try:
            # 读取原始内容
            with open(mmd_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取关键信息
            extracted_content = extract_key_info_from_paper(content)
            
            # 保持相同的目录结构
            rel_path = os.path.relpath(mmd_file, input_dir)
            output_file = os.path.join(output_dir, rel_path)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 写入提取的内容
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_content)
                
        except Exception as e:
            print(f"处理文件 {mmd_file} 时出错: {str(e)}")

def extract_key_info_from_paper(content, max_tokens=30000):
    """Extract key information from papers"""
    # 提取标题
    title = "Unknown Title"
    
    # 尝试匹配标准Markdown标题格式
    title_match = re.search(r'^#\s+(.*?)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
    else:
        # 尝试匹配双星号格式的标题
        title_match = re.search(r'^\*\*(.*?)\*\*\s*$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
    
    # 查找所有章节
    sections = {}
    
    # 首先尝试匹配标准Markdown章节格式
    section_matches = re.findall(r'(?:^|\n)(#{1,3}\s+.*?)(?=\n)(.*?)(?=\n#{1,3}\s+|\Z)', content, re.DOTALL)
    
    # 如果没有找到标准章节，尝试匹配双星号格式的章节
    if not section_matches:
        # 匹配双星号格式的章节标题和内容
        section_matches = re.findall(r'(?:^|\n)(\*\*([^*:]+)(?::|)\*\*)(?=\n)(.*?)(?=\n\*\*[^*]+\*\*|\Z)', content, re.DOTALL)
        if section_matches:
            # 转换匹配结果格式以与标准格式兼容
            section_matches = [(match[0], match[2]) for match in section_matches]
    
    # 如果仍然没有找到任何章节，可能是格式问题，尝试直接使用整个内容
    if not section_matches:
        # 保留标题，然后添加剩余内容
        title_line = None
        if re.search(r'^#\s+.*?$', content, re.MULTILINE):
            title_line = re.search(r'^#\s+.*?$', content, re.MULTILINE)
        elif re.search(r'^\*\*.*?\*\*\s*$', content, re.MULTILINE):
            title_line = re.search(r'^\*\*.*?\*\*\s*$', content, re.MULTILINE)
        
        if title_line:
            remaining_content = content.replace(title_line.group(0), '', 1).strip()
            if remaining_content:
                sections["Content"] = clean_section_content(remaining_content)
    else:
        for header, content_text in section_matches:
            # 处理标准Markdown格式
            if header.startswith('#'):
                header_text = header.strip('# \n')
            # 处理双星号格式
            else:
                header_text = header.strip('* \n')
                # 移除可能的冒号
                header_text = re.sub(r':\s*$', '', header_text)
            
            # 跳过参考文献和致谢章节
            if re.search(r'reference|bibliography|acknowledgement|acknowledgment', header_text, re.IGNORECASE):
                continue
                
            # 跳过与标题相同的章节标题（解决标题重复问题）
            if header_text.lower() == title.lower():
                continue
                
            sections[header_text] = clean_section_content(content_text.strip())
    
    # 提取图表信息
    figure_captions = extract_figure_captions(content)
    table_captions = extract_table_captions(content)
    
    # 组合提取的信息
    combined_text = f"# {title}\n\n"
    
    # 添加所有章节（除了参考文献和与标题重复的章节）
    for header, section_content in sections.items():
        if section_content.strip():  # 只添加非空章节
            combined_text += f"## {header}\n{section_content}\n\n"
    
    # 添加图表注释
    if figure_captions or table_captions:
        combined_text += "## Figures and Tables\n"
        
        if figure_captions:
            combined_text += "### Figures\n"
            combined_text += figure_captions + "\n\n"
            
        if table_captions:
            combined_text += "### Tables\n"
            combined_text += table_captions + "\n\n"
    
    # 最终清理
    combined_text = final_cleanup(combined_text)
    
    # 如果结果只有标题，尝试使用原始内容
    if combined_text.strip() == f"# {title}":
        # 移除参考文献部分
        cleaned_content = re.sub(r'(?:^|\n)#{1,3}\s+(?:References|Bibliography)[\s\S]*$', '', content, flags=re.IGNORECASE)
        cleaned_content = re.sub(r'(?:^|\n)\*\*(?:References|Bibliography)\*\*[\s\S]*$', '', cleaned_content, flags=re.IGNORECASE)
        # 清理内容
        cleaned_content = clean_section_content(cleaned_content)
        # 最终清理
        cleaned_content = final_cleanup(cleaned_content)
        return cleaned_content.strip()
    
    return combined_text.strip()

def clean_section_content(content):
    """清理章节内容"""
    # 移除引用标记 [1], [2,3], etc.
    content = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', content)
    
    # 移除表格、算法和数学公式等技术细节，但保留文本内容
    content = remove_technical_details(content)
    
    # 通用清理
    content = re.sub(r'\n{3,}', '\n\n', content)  # 移除多余空行
    content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)  # 移除行首空格
    
    # 移除脚注
    content = re.sub(r'Footnote \d+:.*?\n', '', content)
    
    return content.strip()

def remove_technical_details(text):
    """移除技术细节，但保留数学公式"""
    # 移除算法伪代码，但保留内部内容
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    for block in code_blocks:
        if re.search(r'import|def|class|for|while|if|return', block, re.IGNORECASE):
            text = text.replace(block, '')
    
    # 保留数学公式 - 注释掉之前移除公式的代码
    # text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    # text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)
    # text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    # text = re.sub(r'\$.*?\$', '', text)
    
    # 移除表格内容（但保留表格注释）
    text = re.sub(r'\\begin\{table\}[\s\S]*?\\end\{table\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{tabular\}[\s\S]*?\\end\{tabular\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\n\|.*\|.*\n', '\n', text)  # 移除Markdown表格
    
    # 移除图片内容（但保留图片注释）
    text = re.sub(r'\\begin\{figure\}[\s\S]*?\\end\{figure\}', '', text, flags=re.DOTALL)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # 移除Markdown图片
    
    return text

def extract_figure_captions(text):
    """提取图片注释"""
    captions = []
    
    # 提取图片标题
    figure_captions = re.findall(r'(?:Figure|Fig\.?)\s+\d+[.:]\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
    
    # 提取图片标题（LaTeX格式）
    latex_captions = re.findall(r'\\caption\{(.*?)\}', text)
    
    # 提取Markdown图片标题
    md_captions = re.findall(r'!\[(.*?)\]\(.*?\)', text)
    
    # 合并所有图片注释
    all_captions = figure_captions + latex_captions + md_captions
    
    # 去重并格式化
    unique_captions = []
    for caption in all_captions:
        caption = caption.strip()
        if caption and caption not in unique_captions:
            unique_captions.append(caption)
    
    # 组合所有图片注释
    if unique_captions:
        return "\n".join([f"- {caption}" for caption in unique_captions])
    return ""

def extract_table_captions(text):
    """提取表格注释"""
    captions = []
    
    # 提取表格标题
    table_captions = re.findall(r'Table\s+\d+[.:]\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
    
    # 提取表格标题（LaTeX格式）
    latex_captions = re.findall(r'\\caption\{(.*?)\}', text)
    
    # 合并所有表格注释
    all_captions = table_captions + latex_captions
    
    # 去重并格式化
    unique_captions = []
    for caption in all_captions:
        caption = caption.strip()
        # 移除表格标题中的数字引用
        caption = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', caption)
        if caption and caption not in unique_captions:
            unique_captions.append(caption)
    
    # 组合所有表格注释
    if unique_captions:
        return "\n".join([f"- {caption}" for caption in unique_captions])
    return ""

def final_cleanup(text):
    """最终清理文本"""
    # 移除多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除空章节
    text = re.sub(r'## [^\n]+\n\s*(?=\n##|$)', '', text, flags=re.DOTALL)
    
    # 移除参考文献部分
    text = re.sub(r'## (?:[Rr]eference|[Rr]eferences|[Bb]ibliography)(?:\s+[^\n]+)?\n[\s\S]*?(?=\n##|$)', '\n\n', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(?:[Rr]eference|[Rr]eferences|[Bb]ibliography)\*\*(?:\s+[^\n]+)?\n[\s\S]*?(?=\n\*\*|\Z)', '\n\n', text, flags=re.DOTALL)
    
    # 移除致谢部分
    text = re.sub(r'## (?:[Aa]cknowledg(?:e)?ment(?:s)?)(?:\s+[^\n]+)?\n[\s\S]*?(?=\n##|$)', '\n\n', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(?:[Aa]cknowledg(?:e)?ment(?:s)?)\*\*(?:\s+[^\n]+)?\n[\s\S]*?(?=\n\*\*|\Z)', '\n\n', text, flags=re.DOTALL)
    
    # 移除LaTeX残留标记
    text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', '', text)
    
    # 移除重复的章节标题
    text = re.sub(r'(## [^\n]+)\n\s*\1', r'\1', text)
    
    # 提取文档标题
    title_match = re.search(r'^# (.*?)$', text, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        # 移除与文档标题相同的章节标题及其内容
        text = re.sub(r'## ' + re.escape(title) + r'\n.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
    
    return text.strip()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "data", "parsed_texts")
    output_dir = os.path.join(base_dir, "data", "extracted_texts")
    
    process_all_papers(input_dir, output_dir)
    print(f"Processing complete, extracted content saved to {output_dir}")