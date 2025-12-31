from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("yolov8n.pt")  # 轻量版 YOLO

seq_len = 30  # 每段视频截取多少帧

# 原始视频目录
video_dirs = {
    "violence": "data/raw_videos/violence",
    "nonviolence": "data/raw_videos/nonviolence"
}

# 输出序列目录
save_root = "data/sequences"
os.makedirs(save_root, exist_ok=True)

def extract_person_sequence(video_path, save_path, seq_len=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        person_boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # COCO class 0 = person
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                person_boxes.append([x1, y1, x2, y2])

        if len(person_boxes) > 0:
            person = max(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            frames.append(person)

        if len(frames) == seq_len:
            break

    cap.release()
    frames = np.array(frames)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, frames)
    print(f"✔ 已保存人体序列: {save_path}")

# 遍历整个数据集
for label, dir_path in video_dirs.items():
    save_dir = os.path.join(save_root, label)
    os.makedirs(save_dir, exist_ok=True)

    for file in os.listdir(dir_path):
        if file.endswith((".mp4", ".avi")):
            video_path = os.path.join(dir_path, file)
            save_path = os.path.join(save_dir, os.path.splitext(file)[0] + ".npy")
            extract_person_sequence(video_path, save_path, seq_len=seq_len)


# videos_to_fix = [
#     "data/raw_videos/nonviolence/NV_341.mp4",
#     "data/raw_videos/nonviolence/NV_365.mp4",
#     "data/raw_videos/nonviolence/NV_387.mp4",
#     "data/raw_videos/nonviolence/NV_393.mp4",
#     "data/raw_videos/nonviolence/NV_402.mp4",
#     "data/raw_videos/nonviolence/NV_415.mp4",
#     "data/raw_videos/nonviolence/NV_444.mp4",
# ]
#
# for video_path in videos_to_fix:
#     save_path = video_path.replace("raw_videos", "sequences").replace(".mp4", ".npy")
#     extract_person_sequence(video_path, save_path, seq_len=30)
