import os
import numpy as np
import torch

seq_folder = "data/sequences/"
classes = ["violence", "nonviolence"]
seq_len = 30  # LSTM 输入序列长度

X, y = [], []

for cls in classes:
    cls_path = os.path.join(seq_folder, cls)
    if not os.path.exists(cls_path):
        print(f"⚠ 文件夹不存在: {cls_path}")
        continue

    files = [f for f in os.listdir(cls_path) if f.endswith(".npy")]
    print(f"正在处理 {cls} 文件夹，共 {len(files)} 个文件...")

    for f in files:
        path = os.path.join(cls_path, f)
        try:
            seq = np.load(path)
            if seq.size == 0:
                print(f"⚠ 空文件，跳过: {path}")
                continue

            # 补齐或截断到 seq_len
            if len(seq) < seq_len:
                last = seq[-1:]
                seq = np.concatenate([seq, np.repeat(last, seq_len - len(seq), axis=0)], axis=0)
            elif len(seq) > seq_len:
                seq = seq[:seq_len]

            X.append(seq)
            y.append(1 if cls == "violence" else 0)

        except Exception as e:
            print(f"⚠ 加载失败，跳过 {path}, 错误: {e}")

# 检查是否有有效样本
if len(X) == 0:
    raise ValueError("没有找到有效的 .npy 文件，dataset.pt 无法生成！")

# 转为 tensor
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
torch.save((X, y), "data/dataset.pt")

print(f"✔ dataset.pt 已生成，样本数: {len(X)}")
print("y unique labels:", torch.unique(y))
