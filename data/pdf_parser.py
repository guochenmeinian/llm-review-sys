import os
import sys
import subprocess
from tqdm import tqdm
import threading, time

def parse_pdfs(pdf_root_dir, output_root_dir, target_conferences=None):
    """
    Use Nougat to parse PDF files
    """
    tasks = []
    all_conferences = [d for d in os.listdir(pdf_root_dir) if os.path.isdir(os.path.join(pdf_root_dir, d))]
    conferences_to_process = target_conferences if target_conferences else all_conferences
    print(f"将处理以下会议: {conferences_to_process}")

    for conference in conferences_to_process:
        pdf_dir = os.path.join(pdf_root_dir, conference)
        output_dir = os.path.join(output_root_dir, conference)
        if not os.path.isdir(pdf_dir):
            continue
        os.makedirs(output_dir, exist_ok=True)
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                mmd_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}.mmd")
                if not os.path.exists(mmd_file):
                    tasks.append((pdf_path, output_dir))

    for task in tqdm(tasks, desc="解析PDF进度"):
        success = process_single_pdf(task)
        if not success:
            print("⚠️ 某个任务失败，但继续执行下一篇")

def process_single_pdf(args):
    pdf_path, output_dir = args
    print("🚀 开始处理:", pdf_path)
    command = ["nougat", pdf_path, "-o", output_dir, "-m", "0.1.0-base"]
    print("Running command:", command)

    try:
        result = subprocess.run(command)
        print("✅ nougat 退出码:", result.returncode)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.mmd")
        if not os.path.exists(output_file):
            print(f"⚠️ 输出文件未生成: {output_file}")
            return False
        return True
    except Exception as e:
        print(f"❌ 解析失败: {pdf_path} - {e}")
        return False

def main():
    base_dir = os.path.dirname(__file__)
    pdf_root = os.path.join(base_dir, "pdfs")
    output_root = os.path.join(base_dir, "parsed_texts")
    TARGET_CONFERENCES = ['NeurIPS_1']

    parse_pdfs(pdf_root, output_root, target_conferences=TARGET_CONFERENCES)

    print("✅ 所有 PDF 解析任务执行完毕！")
    sys.exit(0)

if __name__ == "__main__":
    main()
