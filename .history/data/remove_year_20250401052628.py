import os
import re

# 设置目录（修改为你的实际路径）
txt_dir = os.path.join("aggregated_reviews")  # 或完整路径如 "C:/project/data/aggregated_reviews"

for filename in os.listdir(txt_dir):
    if filename.endswith(".txt"):
        # 使用正则匹配 {name}_{4位数字}.txt
        match = re.match(r"^(.+)_\d{4}\.txt$", filename)
        if match:
            new_name = match.group(1) + ".txt"  # 提取 name 部分
            old_path = os.path.join(txt_dir, filename)
            new_path = os.path.join(txt_dir, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")