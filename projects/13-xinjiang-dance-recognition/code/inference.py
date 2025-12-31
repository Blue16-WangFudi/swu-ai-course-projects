import cv2, torch, numpy as np, joblib, os, pathlib, argparse, tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# ========== 绝对路径（改成你的实际路径） ==========
BASE_DIR = pathlib.Path(__file__).parent              # 当前脚本所在目录
MODEL_PATH = BASE_DIR / 'result6/best.pt'                     # ← 改成你的实际路径
SCALER_PATH = BASE_DIR / 'minmax_scaler.pkl'          # ← 改成你的实际路径

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)   # 修正警告

# 用你训练时的类名和参数（零重新训练）
from train import SimpleLSTM          # ← 改成你训练时的类名
model = SimpleLSTM(hidden=128, layers=2, dropout=0.2)   # ← 用你训练时的参数
model.load_state_dict(state_dict)
model.eval()
# 加载 scaler（复用训练时的 μ,σ）
scaler = joblib.load(SCALER_PATH)

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def predict_video(video_path, fps_target=30, T=30):
    """
    输入：mp4 路径
    输出：{label: 0/1, prob: float, frames: int}
    """
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(orig_fps / fps_target))

    all_landmarks = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                vec = [pt.x for pt in res.pose_landmarks.landmark] + \
                      [pt.y for pt in res.pose_landmarks.landmark] + \
                      [pt.z for pt in res.pose_landmarks.landmark]
            else:
                vec = [0.0] * 99
            all_landmarks.append(vec)
        idx += 1
    cap.release()

    # 长度对齐到 T 帧（训练时的固定长度）
    if len(all_landmarks) == 0:
        return {'label': 0, 'prob': 0.0, 'frames': 0}

    arr = np.array(all_landmarks, dtype=np.float32)          # (F, 99)
    if arr.shape[0] < T:
        # 线性插值到 T 帧
        idx_new = np.inferp(np.linspace(0, arr.shape[0] - 1, T), np.arange(arr.shape[0]), arr)
        arr = idx_new
    else:
        idx = np.linspace(0, arr.shape[0] - 1, T, dtype=int)
        arr = arr[idx]

    # 复用训练时的标准化
    x_std = scaler.transform(arr)          # (T, 99)
    x_std = torch.from_numpy(x_std).unsqueeze(0).float()  # (1, T, 99)

    # 推理
    with torch.no_grad():
        out = model(x_std)                 # (1, 2)
        prob = torch.softmax(out, dim=1)[0, 1].item()      # 规范概率
    label = 1 if prob > 0.5 else 0
    return {'label': label, 'prob': prob, 'frames': arr.shape[0]}