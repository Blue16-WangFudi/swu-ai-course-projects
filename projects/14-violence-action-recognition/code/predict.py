from ultralytics import YOLO
import cv2
import numpy as np
import torch
from lstm_model import FightLSTM
import os
from PIL import Image, ImageDraw, ImageFont
from collections import deque

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备:", device)

# 加载模型
model_yolo = YOLO("yolov8n.pt")
model_yolo.to(device)

model_lstm = FightLSTM().to(device)
model_lstm.load_state_dict(torch.load("lstm_model_best.pth", map_location=device))
model_lstm.eval()

# 中文绘制函数
def put_chinese_text(frame, text, position=(20, 40), font_size=32, color=(0, 0, 255)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 修改为你系统上的中文字体路径
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame_bgr

# 实时滑动窗口预测
def detect_fight_video(video_path, output_path="output.mp4", seq_len=30):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seq_queue = deque(maxlen=seq_len)  # 滑动窗口

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame)[0]
        persons = []

        # 绘制人体 bbox
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                persons.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 取最大 bbox 作为 LSTM 输入
        if persons:
            largest = max(persons, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            seq_queue.append(largest)

        # 如果序列不足 seq_len，使用最后一帧补齐
        if len(seq_queue) == 0:
            pred_text = "视频中未检测到人"
        else:
            padded_seq = list(seq_queue) + [seq_queue[-1]]*(seq_len - len(seq_queue))
            seq_tensor = torch.tensor([padded_seq], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model_lstm(seq_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                prob = torch.softmax(output, dim=1)[0, pred_idx].item()
                pred_text = f"有暴力冲突! ({prob*100:.1f}%)" if pred_idx == 1 else f"没有暴力冲突 ({prob*100:.1f}%)"

        # 绘制中文预测
        frame = put_chinese_text(frame, f"LSTM预测: {pred_text}", (20, 40))

        out.write(frame)

    cap.release()
    out.release()
    print(f"✔ 视频处理完成，已保存到 {output_path}")

# 测试
if __name__ == "__main__":
    test_video = "data/test_1.mp4"  # 修改为你的视频路径
    output_video = "output/test_output1.mp4"
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    detect_fight_video(test_video, output_video)
