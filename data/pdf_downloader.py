import os
import json
import requests
import time
import random
from tqdm import tqdm

def download_pdfs_from_data(data_dir, pdf_dir, target_conferences=None):
    """
    从爬虫获取的数据中下载PDF文件
    Args:
        data_dir: 爬虫数据目录
        pdf_dir: PDF保存目录
        target_conferences: 指定要处理的会议列表，如果为None则处理所有会议
    """
    # 确保PDF目录存在
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    # 获取所有会议目录
    all_conferences = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # 如果指定了目标会议，则只处理这些会议
    conferences_to_process = target_conferences if target_conferences else all_conferences
    print(f"将处理以下会议: {conferences_to_process}")
    
    # 遍历会议目录
    for conf_dir in conferences_to_process:
        conf_path = os.path.join(data_dir, conf_dir)
        if not os.path.exists(conf_path) or not os.path.isdir(conf_path):
            print(f"会议目录不存在: {conf_dir}")
            continue
        
        # 创建会议PDF目录
        conf_pdf_dir = os.path.join(pdf_dir, conf_dir)
        
        # 如果会议PDF目录已存在，跳过整个会议
        if os.path.exists(conf_pdf_dir):
            print(f"⏩ 跳过已处理的会议: {conf_dir}")
            continue
            
        # 创建新的会议PDF目录
        os.makedirs(conf_pdf_dir)
        
        # 查找results.json文件
        results_file = os.path.join(conf_path, "results.json")
        if not os.path.exists(results_file):
            print(f"跳过没有results.json的目录: {conf_dir}")
            continue
        
        # 加载会议数据
        with open(results_file, 'r', encoding='utf-8') as f:
            try:
                venues = json.load(f)
                print(f"处理会议: {conf_dir}, 包含 {len(venues)} 个venue")
            except json.JSONDecodeError:
                print(f"无法解析JSON文件: {results_file}")
                continue
        
        # 遍历所有venue
        for venue in venues:
            if "papers" not in venue:
                continue
                
            # 遍历所有论文
            for paper in venue["papers"]:
                if "pdf_url" not in paper or not paper["pdf_url"]:
                    continue
                
                # 生成PDF文件名
                paper_id = paper["pdf_url"].split("id=")[-1] if "id=" in paper["pdf_url"] else f"paper_{hash(paper['title'])}"
                pdf_filename = f"{paper_id}.pdf"
                pdf_path = os.path.join(conf_pdf_dir, pdf_filename)
                
                # 如果PDF已存在，跳过下载
                if os.path.exists(pdf_path):
                    print(f"⏩ 跳过已存在的PDF: {conf_dir}/{pdf_filename}")
                    continue
                
                # 下载PDF，添加重试机制
                max_retries = 5
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        # 添加随机延迟，避免请求过于频繁
                        delay = 3 + random.random() * 2  # 3-5秒的随机延迟
                        if retry_count > 0:
                            print(f"第 {retry_count} 次重试，等待 {delay:.1f} 秒...")
                            time.sleep(delay)
                        
                        print(f"下载PDF: {paper['title']} -> {conf_dir}/{pdf_filename}")
                        
                        # 添加请求头，模拟浏览器行为
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
                            'Accept': 'application/pdf',
                            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                            'Referer': 'https://openreview.net/',
                        }
                        
                        response = requests.get(paper["pdf_url"], headers=headers, stream=True, timeout=30)
                        response.raise_for_status()
                        
                        with open(pdf_path, 'wb') as pdf_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                pdf_file.write(chunk)
                        
                        print(f"✅ 成功下载: {conf_dir}/{pdf_filename}")
                        success = True
                        
                        # 成功下载后，添加额外延迟，避免连续请求
                        time.sleep(2 + random.random() * 3)  # 2-5秒的随机延迟
                        
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 429:
                            retry_count += 1
                            wait_time = 10 * (2 ** retry_count)  # 指数退避策略
                            print(f"❗ 请求过多 (429)，等待 {wait_time} 秒后重试...")
                            time.sleep(wait_time)
                        else:
                            print(f"❌ 下载失败 {paper['pdf_url']}: {str(e)}")
                            break
                    except Exception as e:
                        print(f"❌ 下载失败 {paper['pdf_url']}: {str(e)}")
                        retry_count += 1
                        time.sleep(5)
                
                if not success and retry_count >= max_retries:
                    print(f"⚠️ 达到最大重试次数，跳过下载: {conf_dir}/{pdf_filename}")

def main():
    # 设置目录
    base_dir = os.path.join(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "raw_data")
    pdf_dir = os.path.join(base_dir, "pdfs")
    
    # 指定要处理的会议列表，如果为空列表则处理所有会议
    TARGET_CONFERENCES = ['AAAI', 'thecvf']  # 例如: ['ICML', 'NeurIPS', 'ICLR']
    
    # 确保目录存在
    os.makedirs(pdf_dir, exist_ok=True)
    
    # 从爬虫数据下载PDF
    download_pdfs_from_data(data_dir, pdf_dir, TARGET_CONFERENCES)
    
    print("PDF下载完成！")

if __name__ == "__main__":
    main()