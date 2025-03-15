import os
import subprocess
import logging
from multiprocessing import Pool
from tqdm import tqdm

def parse_pdfs(pdf_root_dir, output_root_dir, target_conferences=None, num_processes=4):
    """
    使用Nougat解析PDF文件
    Args:
        target_conferences: 指定要处理的会议列表（如['thecvf', 'ACM']）
        pdf_root_dir: PDF存储根目录（如pdfs/）
        output_root_dir: 解析结果输出根目录（如parsed_mmd/）
        num_processes: 并行进程数
    """
    # 创建日志记录器
    logging.basicConfig(filename=os.path.join(output_root_dir, 'parser.log'), 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 收集所有需要处理的PDF文件
    tasks = []
    # 获取所有会议目录
    all_conferences = [d for d in os.listdir(pdf_root_dir) if os.path.isdir(os.path.join(pdf_root_dir, d))]
    
    # 过滤目标会议
    conferences_to_process = target_conferences if target_conferences else all_conferences
    print(f"将处理以下会议: {conferences_to_process}")

    for conference in conferences_to_process:  # 修改这里
        pdf_dir = os.path.join(pdf_root_dir, conference)
        output_dir = os.path.join(output_root_dir, conference)
        
        if not os.path.isdir(pdf_dir):
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                mmd_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}.mmd")
                
                # 跳过已解析的文件
                if not os.path.exists(mmd_file):
                    tasks.append((pdf_path, output_dir))

    # 使用多进程池处理
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_single_pdf, tasks), 
                 total=len(tasks), 
                 desc="解析PDF进度"))

def process_single_pdf(args):
    pdf_path, output_dir = args
    try:
        # 调用Nougat命令行工具
        subprocess.run([
            'nougat', 
            pdf_path,
            '-o', output_dir,
            '--markdown'  # 确保输出为markdown格式
        ], check=True, capture_output=True)
        
        # 验证输出文件
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.mmd")
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"输出文件 {output_file} 未生成")
            
        return True
    except Exception as e:
        logging.error(f"解析失败: {pdf_path} - {str(e)}")
        return False

def main():
    # 设置目录
    base_dir = os.path.join(os.path.dirname(__file__))
    pdf_root = os.path.join(base_dir, "pdfs")
    output_root = os.path.join(base_dir, "parsed_texts")
    
    # 指定目标会议列表
    # done: ['thecvf', ]
    TARGET_CONFERENCES = ['AAAI']  # 可自定义会议列表
    
    # 使用过滤后的会议列表
    parse_pdfs(pdf_root, output_root, 
              target_conferences=TARGET_CONFERENCES,
              num_processes=1)

if __name__ == "__main__":
    main()