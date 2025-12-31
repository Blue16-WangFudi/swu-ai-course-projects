#!/usr/bin/env python3
# gui.py  → 图形窗口版（零配置）
import cv2, torch, numpy as np, joblib, os, pathlib, argparse, tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
mp_pose = mp.solutions.pose

import joblib
scaler = joblib.load('minmax_scaler.pkl')   # ← 来自你训练时的 scaler

# ========== 加载你的训练产物（零重新训练） ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('best.pt', map_location=device, weights_only=True)   # 修正警告

# 用你训练时的类名和参数（零重新训练）
# 如果类名是 SimpleLSTM，就写 SimpleLSTM；如果是 ShortActionLSTM，就写 ShortActionLSTM
from train import SimpleLSTM          # ← 改成你训练时的类名
model = SimpleLSTM(hidden=128, layers=2, dropout=0.2)   # ← 用你训练时的参数
model.load_state_dict(state_dict)
model.eval()


# ========== 核心推理函数 ==========
def predict_video(video_path, fps_target=30, T=30):
    """输入：mp4 路径，输出：{label, prob, frames}"""
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(orig_fps / fps_target))

    all_landmarks = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_pose.process(rgb)
            if res.pose_landmarks:
                vec = [pt.x for pt in res.pose_landmarks.landmark] + \
                      [pt.y for pt in res.pose_landmarks.landmark] + \
                      [pt.z for pt in res.pose_landmarks.landmark]
            else:
                vec = [0.0] * 99
            all_landmarks.append(vec)
        idx += 1
    cap.release()

    if len(all_landmarks) == 0:
        return {'label': 0, 'prob': 0.0, 'frames': 0}

    arr = np.array(all_landmarks, dtype=np.float32)
    # 对齐到 T 帧（训练时的固定长度）
    if arr.shape[0] < T:
        idx_new = np.linspace(0, arr.shape[0] - 1, T, dtype=int)
        arr = arr[idx_new]
    else:
        idx = np.linspace(0, arr.shape[0] - 1, T, dtype=int)
        arr = arr[idx]

    # 复用训练时的标准化
    x_std = scaler.transform(arr)          # (T, 99)
    x_std = torch.from_numpy(x_std).unsqueeze(0).float()  # (1, T, 99)

    with torch.no_grad():
        out = model(x_std)                 # (1, 2)
        prob = torch.softmax(out, dim=1)[0, 1].item()      # 规范概率
    label = 1 if prob > 0.5 else 0
    return {'label': label, 'prob': prob, 'frames': arr.shape[0]}


# ========== GUI ==========
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("新疆舞扭脖子检测")
        self.geometry("500x300")
        self.resizable(False, False)

        # 文件路径
        self.file_path = tk.StringVar()

        # 布局
        tk.Label(self, text="请选择 MP4 文件：", font=("Arial", 14)).pack(pady=10)
        tk.Entry(self, textvariable=self.file_path, width=50, state='readonly').pack(pady=5)
        tk.Button(self, text="浏览...", command=self.browse).pack(pady=5)
        tk.Button(self, text="开始分析", command=self.analyze).pack(pady=10)
        self.result_label = tk.Label(self, text="", font=("Arial", 12), fg="blue")
        self.result_label.pack(pady=10)

    def browse(self):
        file = filedialog.askopenfilename(title="选择 MP4", filetypes=[("MP4 files", "*.mp4")])
        if file:
            self.file_path.set(file)

    def analyze(self):
        path = self.file_path.get()
        if not path:
            messagebox.showerror("错误", "请先选择文件！")
            return
        self.result_label.config(text="分析中…", fg="orange")
        self.update_idletasks()

        try:
            result = predict_video(path)
            label_str = "规范" if result['label'] == 1 else "不规范"
            prob = result['prob'] * 100
            self.result_label.config(
                text=f"结果：{label_str}\n置信度：{prob:.1f}%\n帧数：{result['frames']}",
                fg="green" if result['label'] == 1 else "red"
            )
        except Exception as e:
            self.result_label.config(text=f"失败：{e}", fg="red")


if __name__ == '__main__':
    app = App()
    app.mainloop()