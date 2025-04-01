import os

txt_dir = "aggregated_reviews"
seen_ids = set()

with os.scandir(txt_dir) as entries:
    for entry in entries:
        if entry.name.endswith(".txt"):
            file_id = entry.name
            
            if file_id in seen_ids:
                os.remove(entry.path)
                print(f"🗑️ 已删除重复文件: {entry.name}")
            else:
                seen_ids.add(file_id)
                print(f"✅ 保留文件: {entry.name}")