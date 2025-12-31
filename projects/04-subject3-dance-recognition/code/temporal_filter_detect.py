import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from collections import deque

# ===================== 核心配置（仅保留必要参数） =====================
CONFIG = {
    "yolov5_model_path": "runs/train/KEMU3_2/weights/best.pt",  # 你的模型路径
    "video_path": "data/videos/other1.mp4",  # 测试视频
    "output_video_path": "output.mp4",  # 输出视频
    "conf_thres": 0.75,  # YOLO置信度阈值
    "iou_thres": 0.45,  # NMS阈值
    "device": "0",  # 0=GPU，cpu=CPU
    "imgsz": (640, 640),  # 推理尺寸
    "frame_window": 15,  # 统计最近15帧
    "detect_threshold": 11,  # 15帧中至少7帧检测到才显示框
}

# ===================== 简化版时序过滤 =====================
class SimpleTemporalFilter:
    def __init__(self, frame_window, detect_threshold):
        self.frame_window = frame_window  # 统计窗口大小
        self.detect_threshold = detect_threshold  # 检测次数阈值
        self.detect_queue = deque(maxlen=frame_window)  # 全局队列：存储最近15帧的检测结果（True/False）

    def update(self, is_detected):
        """更新队列：传入当前帧是否检测到单人目标"""
        self.detect_queue.append(is_detected)

    def is_valid(self):
        """判断是否满足阈值：10帧中至少7帧检测到"""
        if len(self.detect_queue) < self.frame_window:
            return False  # 队列未满时暂不判定（避免前几帧误判）
        return sum(self.detect_queue) >= self.detect_threshold

# ===================== YOLOv5推理 =====================
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def init_yolov5_model():
    """初始化YOLO模型"""
    device = select_device(CONFIG["device"])
    model = DetectMultiBackend(CONFIG["yolov5_model_path"], device=device, fp16=False)
    imgsz = check_img_size(CONFIG["imgsz"], s=model.stride)
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    model.warmup(imgsz=(1, 3, imgsz[0], imgsz[1]))
    return model, device, imgsz

def detect_single_frame(model, device, imgsz, frame):
    """单帧检测：仅返回单人目标框"""
    # 预处理
    img = letterbox(frame, imgsz, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]

    # 推理+NMS
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, CONFIG["conf_thres"], CONFIG["iou_thres"], classes=[0], agnostic=False)

    # 后处理：仅取第一个框（单人检测）
    det = pred[0]
    if det is not None and len(det):
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
        # 仅返回第一个框（单人场景）
        return det[0][:5].cpu().numpy()  # [x1,y1,x2,y2,conf]
    return None

# ===================== 主处理函数（核心逻辑） =====================
def main():
    # 初始化
    model, device, imgsz = init_yolov5_model()
    temporal_filter = SimpleTemporalFilter(CONFIG["frame_window"], CONFIG["detect_threshold"])

    # 打开视频
    cap = cv2.VideoCapture(CONFIG["video_path"])
    assert cap.isOpened(), f"无法打开视频：{CONFIG['video_path']}"
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(CONFIG["output_video_path"], fourcc, fps, (w, h))

    # 逐帧处理
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 单帧检测
        det_box = detect_single_frame(model, device, imgsz, frame)
        is_detected = det_box is not None  # 当前帧是否检测到

        # 2. 更新时序过滤队列
        temporal_filter.update(is_detected)

        # 3. 仅当满足15帧阈值时，绘制框
        if temporal_filter.is_valid() and det_box is not None:
            x1, y1, x2, y2, conf = det_box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 绘制框（无ID，仅显示置信度）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf:{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4. 写入视频+预览
        out.write(frame)
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成！输出视频：{os.path.abspath(CONFIG['output_video_path'])}")

if __name__ == "__main__":
    main()
