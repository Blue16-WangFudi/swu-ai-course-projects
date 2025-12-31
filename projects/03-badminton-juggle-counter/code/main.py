import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import tkinter as tk
from tkinter import filedialog
import os
import traceback
from PIL import Image, ImageDraw, ImageFont
import ctypes

# --- Windows DPI 缩放修复 ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class BadmintonJugglingApp:
    def __init__(self, model_path):
        # --- 1. 初始化模型 ---
        print(f"正在加载模型: {model_path} ...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return

        # --- 2. 核心变量 ---
        self.ball_history = deque(maxlen=20)
        self.last_center = None
        self.last_velocity = (0, 0)  # (vx, vy) 平滑后的速度

        # 新增：防抖动与静止过滤变量
        self.static_frame_count = 0
        self.is_tracking = False

        # 掉球与重置相关
        self.last_seen_time = time.time()
        self.floor_dwell_frames = 0

        # 计数逻辑
        self.is_counting = False
        self.current_count = 0
        self.history_records = []
        self.state = "IDLE"
        self.cooldown_frames = 0

        # --- 3. 核心算法参数 (可在此调整) ---
        self.algo_params = {
            # 动态矩形搜索参数
            'base_buffer': 80,
            'long_factor': 1.5,
            'lat_factor': 0.3,

            # 静止过滤参数
            'static_dist_thresh': 5,
            'max_static_frames': 15,

            # 计数灵敏度参数 (新加)
            'impulse_threshold': -15.0,
            'min_rising_speed': -2.0,

            # 其他
            'drop_timeout': 3.0,
            'floor_y_ratio': 0.85,
        }

        # --- 4. 视频与UI参数 ---
        self.cap = None
        self.running = True
        self.is_video_file = False
        self.display_height = 720
        self.display_width = 1280
        self.ui_width = 320

        # 字体与UI
        self.font_path = "C:/Windows/Fonts/msyh.ttc"
        if not os.path.exists(self.font_path):
            self.font_path = "C:/Windows/Fonts/simhei.ttf"

        self.window_name = "Badminton AI Counter Pro"
        self.buttons = {}
        self.root = tk.Tk()
        self.root.withdraw()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280 + self.ui_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.load_source(0)

    def load_source(self, source):
        if self.cap is not None:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise Exception("无法打开源")
            self.is_video_file = isinstance(source, str)

            orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ratio = self.display_height / orig_h
            self.display_width = int(orig_w * ratio)

            cv2.resizeWindow(self.window_name, self.display_width + self.ui_width, self.display_height)
            self.manual_reset()
            print(f"✅ 加载成功: {self.display_width}x{self.display_height}")
        except Exception as e:
            print(f"❌ 错误: {e}")
            self.cap = None

    def manual_reset(self):
        if self.current_count > 0:
            self.history_records.append(self.current_count)
        self.current_count = 0
        self._reset_physics()
        self.is_counting = False

    def auto_reset_drop(self):
        if self.current_count > 0:
            print(f"📉 检测到掉球/静止，记录成绩: {self.current_count}")
            self.history_records.append(self.current_count)
        self.current_count = 0
        self._reset_physics()
        self.is_counting = True

    def _reset_physics(self):
        self.ball_history.clear()
        self.last_center = None
        self.last_velocity = (0, 0)
        self.state = "IDLE"
        self.floor_dwell_frames = 0
        self.static_frame_count = 0
        self.is_tracking = False
        self.last_seen_time = time.time()

    def select_video_file(self):
        self.is_counting = False
        try:
            self.root.deiconify()
            self.root.attributes('-topmost', True)
            file_path = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")]
            )
            self.root.withdraw()
            if file_path:
                self.load_source(file_path)
        except Exception as e:
            print(f"UI Error: {e}")

    def get_center(self, box):
        return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

    # --- 动态矩形验证 ---
    def validate_detection_optimized(self, new_center, debug_frame=None):
        if self.last_center is None:
            return True, None

        speed = np.sqrt(self.last_velocity[0] ** 2 + self.last_velocity[1] ** 2)

        pred_x = self.last_center[0] + self.last_velocity[0]
        pred_y = self.last_center[1] + self.last_velocity[1]

        dx = new_center[0] - pred_x
        dy = new_center[1] - pred_y
        dist_sq = dx ** 2 + dy ** 2

        if dist_sq < self.algo_params['base_buffer'] ** 2:
            return True, None

        if speed < 2.0:
            limit = self.algo_params['base_buffer'] * 1.5
            return dist_sq < limit ** 2, None

        u_x = self.last_velocity[0] / speed
        u_y = self.last_velocity[1] / speed

        d_parallel = abs(dx * u_x + dy * u_y)
        d_normal = abs(dx * (-u_y) + dy * u_x)

        limit_long = speed * self.algo_params['long_factor'] + self.algo_params['base_buffer']
        limit_lat = speed * self.algo_params['lat_factor'] + (self.algo_params['base_buffer'] * 0.6)

        if debug_frame is not None:
            p_end = (int(pred_x + u_x * 50), int(pred_y + u_y * 50))
            cv2.line(debug_frame, (int(pred_x), int(pred_y)), p_end, (0, 0, 255), 1)

        if d_parallel <= limit_long and d_normal <= limit_lat:
            return True, None
        else:
            return False, (d_parallel, d_normal, limit_long, limit_lat)

    def process_physics_and_count(self, raw_ball_boxes, frame_debug=None):
        current_time = time.time()
        best_center = None
        min_score = float('inf')

        # --- A. 筛选最佳目标 ---
        if raw_ball_boxes:
            for box in raw_ball_boxes:
                c = self.get_center(box)
                is_valid, _ = self.validate_detection_optimized(c, frame_debug)
                if is_valid:
                    if self.last_center is not None:
                        pred_x = self.last_center[0] + self.last_velocity[0]
                        pred_y = self.last_center[1] + self.last_velocity[1]
                        dist = np.sqrt((c[0] - pred_x) ** 2 + (c[1] - pred_y) ** 2)
                    else:
                        dist = 0
                    if dist < min_score:
                        min_score = dist
                        best_center = c
                else:
                    if frame_debug is not None:
                        cv2.circle(frame_debug, c, 5, (0, 0, 150), -1)

        # --- B. 静止过滤 ---
        if best_center is not None:
            if self.last_center is not None:
                move_dist = np.sqrt((best_center[0] - self.last_center[0]) ** 2 +
                                    (best_center[1] - self.last_center[1]) ** 2)
                if move_dist < self.algo_params['static_dist_thresh']:
                    self.static_frame_count += 1
                else:
                    self.static_frame_count = 0
            else:
                self.static_frame_count = 0

            if self.static_frame_count > self.algo_params['max_static_frames']:
                # print(f"⚠️ 忽略静止物体")
                best_center = None
                self.is_tracking = False

        # --- C. 掉球判定 ---
        is_dropped = False
        if current_time - self.last_seen_time > self.algo_params['drop_timeout']:
            if self.is_counting and self.current_count > 0:
                is_dropped = True

        if best_center is not None:
            self.last_seen_time = current_time
            self.is_tracking = True

            floor_line = self.display_height * self.algo_params['floor_y_ratio']
            if best_center[1] > floor_line:
                if self.last_velocity[1] > -2:
                    self.floor_dwell_frames += 1
                else:
                    self.floor_dwell_frames = 0
            else:
                self.floor_dwell_frames = 0

            if self.floor_dwell_frames > 45:
                if self.is_counting and self.current_count > 0:
                    is_dropped = True
                else:
                    self._reset_physics()
        elif self.is_tracking:
            self.last_velocity = (self.last_velocity[0] * 0.98, self.last_velocity[1] * 0.98)

        if is_dropped:
            self.auto_reset_drop()
            return

        # --- D. 物理更新与计数逻辑 (重点修改区域) ---
        if best_center is not None:
            # 1. 计算瞬时速度 (不做平滑，反应最灵敏)
            if self.last_center is not None:
                inst_vx = best_center[0] - self.last_center[0]
                inst_vy = best_center[1] - self.last_center[1]

                # 2. 计算 Y轴加速度 (当前瞬时速度 - 上一帧平滑速度)
                # 如果 acceleration_y 为负大值，说明受到了向上的强力
                acceleration_y = inst_vy - self.last_velocity[1]

                # 更新平滑速度用于下一帧参考
                self.last_velocity = (
                    0.5 * self.last_velocity[0] + 0.5 * inst_vx,
                    0.5 * self.last_velocity[1] + 0.5 * inst_vy
                )
            else:
                inst_vy = 0
                acceleration_y = 0
                self.last_velocity = (0, 0)

            self.last_center = best_center
            self.ball_history.append(best_center)

            # --- E. 计数判定逻辑 ---
            if self.cooldown_frames > 0: self.cooldown_frames -= 1

            # 判定条件 1: 传统状态机 (下落 -> 上升)
            # 适用于常规颠球，速度变化平缓
            avg_vy = self.last_velocity[1]
            if avg_vy > 2.0:
                self.state = "FALLING"

            is_standard_hit = (self.state == "FALLING" and avg_vy < self.algo_params['min_rising_speed'])

            # 判定条件 2: 冲量/加速度判定 (Rising -> Accelerated Rising)
            # 适用于发球抢攻：球在上升或刚停滞时，被用力向上打
            # 条件：向上的加速度极大 (<-15)，且击球后球确实是向上走的 (< -2)
            is_impulse_hit = (acceleration_y < self.algo_params['impulse_threshold'] and
                              inst_vy < self.algo_params['min_rising_speed'])

            # 综合判定
            if (is_standard_hit or is_impulse_hit) and self.cooldown_frames == 0:
                if self.is_counting:
                    self.current_count += 1
                    print(f"🏸 击球! Acc:{acceleration_y:.1f} Vy:{inst_vy:.1f}")

                self.state = "RISING"
                self.cooldown_frames = 10  # 冷却防止重复计数

            elif avg_vy < self.algo_params['min_rising_speed']:
                self.state = "RISING"  # 保持状态

    def draw_ui_optimized(self, frame_resized):
        h, w = frame_resized.shape[:2]
        sidebar = np.zeros((h, self.ui_width, 3), dtype=np.uint8)
        sidebar[:] = (40, 40, 40)
        final_img_cv = np.hstack((frame_resized, sidebar))

        base_x = w + 20
        btn_start_y = 320
        btn_h = 50
        btn_gap = 20
        btn_w = self.ui_width - 40

        btn_data = [
            ("开始 / 暂停", (50, 180, 50) if not self.is_counting else (0, 140, 255), "START"),
            ("手动归零", (180, 100, 0), "RESET"),
            ("加载视频", (100, 100, 100), "LOAD"),
            ("退出", (50, 50, 200), "EXIT")
        ]

        self.buttons = {}
        for i, (txt, color, key) in enumerate(btn_data):
            y1 = btn_start_y + i * (btn_h + btn_gap)
            y2 = y1 + btn_h
            x1 = base_x
            x2 = base_x + btn_w
            cv2.rectangle(final_img_cv, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(final_img_cv, (x1, y1), (x2, y2), (200, 200, 200), 1)
            self.buttons[key] = (x1, y1, x2, y2)

        floor_y = int(self.display_height * self.algo_params['floor_y_ratio'])
        cv2.line(final_img_cv, (0, floor_y), (self.display_width, floor_y), (0, 255, 255), 1)

        img_pil = Image.fromarray(cv2.cvtColor(final_img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        font_L = ImageFont.truetype(self.font_path, 28, encoding="utf-8")
        font_M = ImageFont.truetype(self.font_path, 22, encoding="utf-8")
        font_S = ImageFont.truetype(self.font_path, 18, encoding="utf-8")
        font_Num = ImageFont.truetype(self.font_path, 90, encoding="utf-8")

        draw.text((base_x, 30), "羽毛球颠球Pro", fill=(255, 215, 0), font=font_L)
        status_txt = "状态: 🟢 计数中" if self.is_counting else "状态: ⏸️ 已暂停"
        draw.text((base_x, 140), status_txt, fill=(50, 205, 50) if self.is_counting else (200, 200, 200), font=font_M)
        draw.text((base_x + 60, 180), str(self.current_count), fill=(50, 255, 50), font=font_Num)
        draw.text((base_x + 220, 260), "个", fill=(150, 150, 150), font=font_M)

        for i, (txt, color, key) in enumerate(btn_data):
            y1 = btn_start_y + i * (btn_h + btn_gap)
            txt_w = len(txt) * 22
            off_x = (btn_w - txt_w) / 2 + 10
            draw.text((base_x + int(off_x), y1 + 10), txt, fill=(255, 255, 255), font=font_M)

        draw.text((base_x, 600), "最近成绩:", fill=(180, 180, 180), font=font_M)
        for i, score in enumerate(self.history_records[-3:]):
            rec_txt = f"#{len(self.history_records) - 2 + i}:  {score} 个"
            draw.text((base_x, 630 + i * 25), rec_txt, fill=(150, 150, 150), font=font_S)

        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, (bx1, by1, bx2, by2) in self.buttons.items():
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.handle_click(key)
                    break

    def handle_click(self, key):
        if key == "START":
            self.is_counting = not self.is_counting
            if self.is_counting:
                self.last_seen_time = time.time()
        elif key == "RESET":
            self.manual_reset()
        elif key == "LOAD":
            self.select_video_file()
        elif key == "EXIT":
            self.running = False

    def run(self):
        print(">>> 系统启动")
        while self.running:
            try:
                if self.cap is None:
                    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
                    final = self.draw_ui_optimized(dummy)
                    cv2.imshow(self.window_name, final)
                    if cv2.waitKey(10) & 0xFF == 27: break
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    if self.is_video_file:
                        if self.current_count > 0: self.history_records.append(self.current_count)
                        self.current_count = 0
                        self._reset_physics()
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                frame_resized = cv2.resize(frame, (self.display_width, self.display_height))
                results = self.model.predict(frame_resized, verbose=False, conf=0.3, iou=0.5)

                candidate_boxes = []
                if results[0].boxes.id is not None or len(results[0].boxes) > 0:
                    for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                              results[0].boxes.cls.cpu().numpy(),
                                              results[0].boxes.conf.cpu().numpy()):
                        if int(cls) == 0:
                            candidate_boxes.append(box)

                self.process_physics_and_count(candidate_boxes, frame_resized)

                if self.last_center:
                    cv2.circle(frame_resized, self.last_center, 6, (0, 255, 0), -1)

                pts = list(self.ball_history)
                for i in range(1, len(pts)):
                    if pts[i - 1] and pts[i]:
                        thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
                        cv2.line(frame_resized, pts[i - 1], pts[i], (0, 255, 0), thickness)

                final_ui = self.draw_ui_optimized(frame_resized)
                cv2.imshow(self.window_name, final_ui)

                if cv2.waitKey(1) & 0xFF == 27:
                    self.running = False

            except Exception as e:
                print("运行时错误:")
                traceback.print_exc()
                time.sleep(1)

        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        try:
            self.root.destroy()
        except:
            pass
        print(">>> 安全退出")


if __name__ == '__main__':
    WEIGHTS = r"D:\pythonProject30\runs\train\badminton_nano_p2_12803\weights\best.pt"
    if os.path.exists(WEIGHTS):
        app = BadmintonJugglingApp(WEIGHTS)
        app.run()
    else:
        print(f"❌ 找不到模型文件: {WEIGHTS}")