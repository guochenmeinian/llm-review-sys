import os

base_dir = "."  # 替换为你的实际路径
txt_dir = os.path.join(base_dir, "data", "aggregated_reviews")

def delete_all_txt_txt(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt.txt"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted: {filename}")

delete_all_txt_txt(txt_dir)
print("All .txt.txt files have been deleted.")