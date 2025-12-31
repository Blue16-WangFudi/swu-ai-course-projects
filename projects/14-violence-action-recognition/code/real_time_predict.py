import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
from lstm_model import FightLSTM
from PIL import Image, ImageDraw, ImageFont

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备:", device)

# 加载模型
model_yolo = YOLO("yolov8n.pt").to(device)

model_lstm = FightLSTM().to(device)
model_lstm.load_state_dict(torch.load("lstm_model_best.pth", map_location=device))
model_lstm.eval()

# 中文显示函数
def put_chinese_text(frame, text, pos=(20, 40), color=(0, 0, 255), size=32):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 摄像头实时检测
def detect_fight_camera(seq_len=30):
    cap = cv2.VideoCapture(0)
    seq_queue = deque(maxlen=seq_len)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame)[0]
        persons = []

        # 获取所有 person bbox
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                persons.append([x1, y1, x2, y2])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 选择最大 bbox
        if persons:
            largest = max(persons, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
            seq_queue.append(largest)

        # LSTM 预测
        if len(seq_queue) == 0:
            txt = "未检测到人"
        else:
            padded = list(seq_queue) + [seq_queue[-1]]*(seq_len - len(seq_queue))
            seq_tensor = torch.tensor([padded], dtype=torch.float32).to(device)

            with torch.no_grad():
                out = model_lstm(seq_tensor)
                pred = torch.argmax(out,1).item()
                prob = torch.softmax(out,1)[0,pred].item()

                txt = f"暴力冲突! ({prob*100:.1f}%)" if pred==1 else f"没有暴力冲突 ({prob*100:.1f}%)"

        frame = put_chinese_text(frame, f"LSTM预测: {txt}", (20, 40))

        cv2.imshow("Fight Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_fight_camera()
