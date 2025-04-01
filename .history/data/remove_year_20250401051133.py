import os
import shutil

def clean_filenames(directory):
    """
    清理目录中的文件名，将"id_year.txt"格式的文件名改为"id.txt"
    
    参数:
    directory (str): 包含文件的目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 只处理.txt文件
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # 提取id部分（假设格式为id_year.txt）
            try:
                id_part = filename.split('_')[0]
                new_filename = f"{id_part}.txt"
                new_file_path = os.path.join(directory, new_filename)
                
                # 重命名文件
                shutil.copy2(file_path, new_file_path)
                print(f"已创建新文件: {new_filename}")
                
            except IndexError:
                print(f"文件 {filename} 格式不符合id_year.txt格式，跳过")

# 使用示例
if __name__ == "__main__":
    # 使用你提供的路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_dir = os.path.join(base_dir, "data", "aggregated_reviews")
    
    clean_filenames(txt_dir)
    print("文件名清理完成")