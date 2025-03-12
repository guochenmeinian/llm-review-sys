from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.checkpoint import get_checkpoint
import torch
import os
import json
from tqdm import tqdm

class PaperParser:
    def __init__(self, checkpoint_path=None):
        """
        初始化 Nougat PDF 解析器
        Args:
            checkpoint_path: 模型检查点路径，如果为 None 则自动下载
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if checkpoint_path is None:
            checkpoint_path = get_checkpoint()
        
        self.model = NougatModel.from_pretrained(checkpoint_path).to(self.device)
        self.model.eval()

    def parse_pdf(self, pdf_path):
        """
        解析单个PDF文件
        Args:
            pdf_path: PDF文件路径
        Returns:
            dict: 包含解析结果的字典
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        dataset = LazyDataset([pdf_path])
        
        parsed_content = []
        with torch.no_grad():
            for sample in dataset:
                image = sample['image'].unsqueeze(0).to(self.device)
                output = self.model.inference(image=image)
                parsed_content.append(output[0])
        
        # 解析输出为结构化数据
        content = '\n'.join(parsed_content)
        return self._structure_content(content)

    def _structure_content(self, content):
        """
        将Markdown格式的内容结构化
        Args:
            content: Nougat输出的Markdown文本
        Returns:
            dict: 结构化的内容
        """
        sections = content.split('\n\n')
        result = {
            'title': '',
            'abstract': '',
            'main_content': '',
            'figures': []
        }
        
        current_section = ''
        for section in sections:
            if section.startswith('# '):  # 标题
                result['title'] = section.replace('# ', '')
            elif section.lower().startswith('abstract'):  # 摘要
                result['abstract'] = section
            elif section.startswith('!['):  # 图表
                result['figures'].append(section)
            else:
                current_section += section + '\n\n'
        
        result['main_content'] = current_section.strip()
        return result

def main():
    # 设置目录
    base_dir = os.path.join(os.path.dirname(__file__))
    pdf_dir = os.path.join(base_dir, "pdfs")
    output_dir = os.path.join(base_dir, "pdfs_parsed")
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化解析器
    parser = PaperParser()
    
    # 处理所有PDF文件
    for pdf_file in tqdm(os.listdir(pdf_dir)):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
            # 检查是否已经解析过
            output_path = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}.json")
            if os.path.exists(output_path):
                print(f"⏩ 跳过已解析的PDF: {pdf_file}")
                continue
                
            try:
                # 解析PDF
                parsed_data = parser.parse_pdf(pdf_path)
                
                # 保存解析结果
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 成功解析: {pdf_file}")
            except Exception as e:
                print(f"❌ 解析 {pdf_file} 时出错: {str(e)}")

if __name__ == "__main__":
    main()