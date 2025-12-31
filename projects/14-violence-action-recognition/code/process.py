import os
import shutil

# 数据集根目录
src_root = "archive/Real Life Violence Dataset"
dst_root = "data/raw_videos"

# 创建目标目录
os.makedirs(os.path.join(dst_root, "violence"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "nonviolence"), exist_ok=True)

# 处理 Violence 文件夹
violence_src = os.path.join(src_root, "Violence")
for file in os.listdir(violence_src):
    if file.endswith((".mp4", ".avi")):
        shutil.copy(os.path.join(violence_src, file),
                    os.path.join(dst_root, "violence", file))

# 处理 NonViolence 文件夹
nonviolence_src = os.path.join(src_root, "NonViolence")
for file in os.listdir(nonviolence_src):
    if file.endswith((".mp4", ".avi")):
        shutil.copy(os.path.join(nonviolence_src, file),
                    os.path.join(dst_root, "nonviolence", file))

print("✔ 数据整理完成！")
