import cv2
import threading
import time
from queue import Queue


class VideoProcessor:
    def __init__(self, video_path, process_frame_func):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.process_frame = process_frame_func  # 处理单帧的函数
        self.running = False
        self.paused = False
        self.frame_queue = Queue(maxsize=10)  # 帧队列，防止处理不及时
        self.result_queue = Queue()  # 结果队列
        self.offside_frames = []  # 记录越位帧
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def start(self):
        self.running = True
        self.paused = False
        # 读取线程
        self.read_thread = threading.Thread(target=self._read_frames)
        self.read_thread.daemon = True
        self.read_thread.start()

        # 处理线程
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

    def _read_frames(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not self.frame_queue.full():
                self.frame_queue.put((self.current_frame, frame))
            else:
                # 队列满时跳过一些帧，防止内存溢出
                time.sleep(0.01)

    def _process_frames(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                frame_num, frame = self.frame_queue.get(timeout=1)
                # 处理帧并获取结果
                result, processed_frame = self.process_frame(frame_num, frame)
                if result is True:  # 检测到越位
                    self.offside_frames.append(frame_num)
                self.result_queue.put((frame_num, processed_frame, result))
                self.frame_queue.task_done()
            except Exception as e:
                continue

    def pause(self):
        self.paused = not self.paused

    def stop(self):
        self.running = False
        if self.read_thread.is_alive():
            self.read_thread.join()
        if self.process_thread.is_alive():
            self.process_thread.join()
        self.cap.release()

    def get_progress(self):
        return (self.current_frame / self.total_frames) * 100 if self.total_frames > 0 else 0

    def get_offside_summary(self):
        """返回越位事件摘要"""
        if not self.offside_frames:
            return "未检测到越位事件"

        # 去重连续帧，只保留起始帧
        unique_events = []
        prev_frame = -10  # 超过10帧间隔认为是不同事件
        for frame in sorted(self.offside_frames):
            if frame - prev_frame > 10:
                unique_events.append(frame)
                prev_frame = frame

        return f"共检测到 {len(unique_events)} 次越位事件，发生在帧: {', '.join(map(str, unique_events))}"