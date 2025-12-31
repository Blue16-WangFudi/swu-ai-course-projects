import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
from collections import deque
import os
import time
import winsound
import json
from datetime import datetime

class BasketballTravelAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # 加载模型
        self.ball_model = YOLO("basketballModel.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")
        
        # 打开视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("=" * 60)
        print(f"视频文件: {video_path}")
        print(f"视频尺寸: {self.frame_width}x{self.frame_height}")
        print(f"帧率: {self.fps:.2f} FPS")
        print(f"总帧数: {self.total_frames}")
        print("=" * 60)
        
        # 关键点索引
        self.body_index = {
            "left_wrist": 10, "right_wrist": 9,
            "left_ankle": 15, "right_ankle": 16,
            "left_knee": 13, "right_knee": 14
        }
        
        # 初始化状态变量
        self.reset_counters()
        
        # 创建输出目录
        self.create_output_directories()
        
        # 初始化输出视频
        self.init_output_video()
        
    def reset_counters(self):
        """重置所有计数器和状态"""
        # 篮球状态
        self.ball_position = None
        self.ball_bbox = None
        self.ball_detected = False
        
        # 持球状态
        self.ball_carrier_id = None  # 持球球员ID
        self.ball_carrier_confidence = 0  # 持球置信度
        self.holding_frames = 0  # 连续持球帧数
        self.min_holding_frames = 5  # 最小持球帧数（用于过滤短暂接触）
        
        # 运球相关
        self.dribble_count = 0
        self.total_dribbles = 0
        self.prev_ball_y = None
        self.prev_delta_y = None
        self.dribble_threshold = 15  # 运球垂直位移阈值
        self.dribble_cooldown = 0  # 运球冷却帧数（避免重复计数）
        
        # 步数相关
        self.step_count = 0
        self.total_steps = 0
        self.carrier_ankle_history = []  # 持球球员脚踝历史位置
        self.step_threshold = 15  # 步数检测阈值（提高以减少误检）
        self.max_history_frames = 10  # 最大历史帧数
        
        # 走步检测
        self.travel_violation = False
        self.travel_start_frame = None
        self.travel_count = 0
        self.travel_frames = []  # 存储走步帧的索引
        
        # 帧计数器
        self.current_frame = 0
        self.start_time = time.time()
        
        # 球员追踪
        self.player_tracking = {}  # 存储每个球员的状态
        
    def create_output_directories(self):
        """创建输出目录"""
        self.output_dir = "analysis_results"
        self.violation_dir = os.path.join(self.output_dir, "violations")
        
        for dir_path in [self.output_dir, self.violation_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def init_output_video(self):
        """初始化输出视频"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_video_path = os.path.join(
            self.output_dir, 
            f"analyzed_{os.path.basename(self.video_path).split('.')[0]}_{timestamp}.mp4"
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path, fourcc, self.fps, 
            (self.frame_width, self.frame_height)
        )
    
    def get_player_keypoints(self, pose_results):
        """获取所有球员的关键点"""
        players = []
        
        try:
            if pose_results[0].keypoints is not None:
                # 获取所有检测到的人的关键点
                if hasattr(pose_results[0].keypoints, 'xy'):
                    all_keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                    confidences = pose_results[0].keypoints.conf.cpu().numpy() if hasattr(pose_results[0].keypoints, 'conf') else None
                    
                    for i, keypoints in enumerate(all_keypoints):
                        if len(keypoints) >= 17:  # COCO关键点模型有17个关键点
                            player_data = {
                                'id': i,
                                'keypoints': keypoints,
                                'confidence': confidences[i] if confidences is not None and i < len(confidences) else None
                            }
                            players.append(player_data)
        except Exception as e:
            print(f"获取球员关键点时出错: {e}")
        
        return players
    
    def detect_ball_carrier(self, ball_pos, players):
        """检测持球球员"""
        if ball_pos is None or len(players) == 0:
            return None, 0
        
        best_carrier = None
        best_confidence = 0
        min_distance = float('inf')
        
        for player in players:
            keypoints = player['keypoints']
            
            # 获取手腕位置
            left_wrist = keypoints[self.body_index["left_wrist"]][:2]
            right_wrist = keypoints[self.body_index["right_wrist"]][:2]
            
            # 计算到篮球的距离
            left_dist = np.linalg.norm(ball_pos - left_wrist)
            right_dist = np.linalg.norm(ball_pos - right_wrist)
            min_dist = min(left_dist, right_dist)
            
            # 计算持球置信度（距离越近，置信度越高）
            distance_threshold = 100  # 像素
            if min_dist < distance_threshold:
                confidence = 1.0 - (min_dist / distance_threshold)
                
                # 考虑球员的姿势：手是否在篮球附近
                # 这里可以添加更多逻辑，比如手是否张开等
                
                if confidence > best_confidence and min_dist < min_distance:
                    best_confidence = confidence
                    min_distance = min_dist
                    best_carrier = player['id']
        
        return best_carrier, best_confidence
    
    def track_ball_carrier(self, carrier_id, carrier_confidence):
        """追踪持球球员"""
        # 如果检测到持球球员
        if carrier_id is not None and carrier_confidence > 0.3:
            # 如果是同一个球员持续持球
            if self.ball_carrier_id == carrier_id:
                self.holding_frames += 1
                # 只有当连续持球超过一定帧数时才确认为持球
                if self.holding_frames >= self.min_holding_frames:
                    return True
            else:
                # 新的持球球员
                self.ball_carrier_id = carrier_id
                self.ball_carrier_confidence = carrier_confidence
                self.holding_frames = 1
                self.carrier_ankle_history = []  # 重置脚踝历史
                print(f"帧 {self.current_frame}: 检测到球员 {carrier_id} 可能持球 (置信度: {carrier_confidence:.2f})")
        else:
            # 没有检测到持球球员
            self.holding_frames = max(0, self.holding_frames - 2)  # 逐渐减少持球计数
            
            # 如果连续多帧没有检测到持球，则重置持球状态
            if self.holding_frames == 0:
                if self.ball_carrier_id is not None:
                    print(f"帧 {self.current_frame}: 球员 {self.ball_carrier_id} 失去持球")
                self.ball_carrier_id = None
                self.ball_carrier_confidence = 0
                self.carrier_ankle_history = []
        
        return self.holding_frames >= self.min_holding_frames
    
    def detect_dribble(self):
        """检测运球动作"""
        if not self.ball_detected or self.ball_position is None:
            return False
        
        current_y = self.ball_position[1]
        
        if self.prev_ball_y is not None:
            delta_y = current_y - self.prev_ball_y
            
            # 减少运球冷却
            if self.dribble_cooldown > 0:
                self.dribble_cooldown -= 1
            
            # 检测运球模式：篮球有明显的上下运动
            if self.prev_delta_y is not None and self.dribble_cooldown == 0:
                # 篮球从上升转为下降（可能触地反弹）
                if self.prev_delta_y > 5 and delta_y < -5:  # 阈值调整
                    # 检查下落幅度是否足够大
                    if abs(delta_y) > self.dribble_threshold:
                        self.dribble_count += 1
                        self.total_dribbles += 1
                        self.dribble_cooldown = 10  # 设置冷却帧数，避免重复计数
                        print(f"帧 {self.current_frame}: 运球检测! 计数: {self.dribble_count}")
                        return True
            
            self.prev_delta_y = delta_y
        
        self.prev_ball_y = current_y
        return False
    
    def count_carrier_steps(self, carrier_keypoints):
        """统计持球球员的步数"""
        if carrier_keypoints is None:
            return 0
        
        try:
            # 获取脚踝位置
            left_ankle = carrier_keypoints[self.body_index["left_ankle"]][:2]
            right_ankle = carrier_keypoints[self.body_index["right_ankle"]][:2]
            
            # 将当前位置添加到历史
            self.carrier_ankle_history.append({
                'frame': self.current_frame,
                'left_ankle': left_ankle,
                'right_ankle': right_ankle
            })
            
            # 保持历史长度
            if len(self.carrier_ankle_history) > self.max_history_frames:
                self.carrier_ankle_history.pop(0)
            
            # 需要至少2帧历史数据才能计算步数
            if len(self.carrier_ankle_history) < 2:
                return 0
            
            new_steps = 0
            
            # 获取当前帧和上一帧
            current = self.carrier_ankle_history[-1]
            prev = self.carrier_ankle_history[-2]
            
            # 计算脚踝移动距离
            left_distance = np.linalg.norm(current['left_ankle'] - prev['left_ankle'])
            right_distance = np.linalg.norm(current['right_ankle'] - prev['right_ankle'])
            
            # 判断哪只脚移动更多
            if left_distance > right_distance and left_distance > self.step_threshold:
                # 检查左脚是否从静止开始移动
                # 我们可以检查更早的历史帧来判断脚是否之前是静止的
                if len(self.carrier_ankle_history) >= 3:
                    prev2 = self.carrier_ankle_history[-3]
                    left_distance_prev = np.linalg.norm(prev['left_ankle'] - prev2['left_ankle'])
                    # 如果之前脚是静止的（移动很小），现在移动了，计为一步
                    if left_distance_prev < self.step_threshold / 3:
                        new_steps = 1
            elif right_distance > left_distance and right_distance > self.step_threshold:
                # 检查右脚是否从静止开始移动
                if len(self.carrier_ankle_history) >= 3:
                    prev2 = self.carrier_ankle_history[-3]
                    right_distance_prev = np.linalg.norm(prev['right_ankle'] - prev2['right_ankle'])
                    if right_distance_prev < self.step_threshold / 3:
                        new_steps = 1
            
            return new_steps
            
        except Exception as e:
            print(f"统计持球球员步数时出错: {e}")
            return 0
    
    def detect_travel_violation(self):
        """检测走步违例"""
        # 篮球规则：持球后，在运球前最多可以走两步
        if self.ball_carrier_id is not None:
            # 检查条件：有持球球员，且该球员步数超过限制且无运球
            if self.step_count > 2 and self.dribble_count == 0:
                return True
        
        return False
    
    def process_frame(self, frame):
        """处理单帧"""
        self.current_frame += 1
        
        # 篮球检测
        self.ball_detected = False
        self.ball_position = None
        self.ball_bbox = None
        
        ball_results = self.ball_model(frame, verbose=False, conf=0.5)
        if ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
            # 取置信度最高的篮球检测
            bbox = ball_results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox[:4]
            ball_x = (x1 + x2) / 2
            ball_y = (y1 + y2) / 2
            self.ball_position = np.array([ball_x, ball_y])
            self.ball_bbox = (int(x1), int(y1), int(x2), int(y2))
            self.ball_detected = True
            
            # 绘制篮球边界框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(ball_x), int(ball_y)), 5, (0, 0, 255), -1)
        
        # 姿态检测
        pose_results = self.pose_model(frame, verbose=False, conf=0.3, max_det=10)  # 最多检测10个人
        pose_annotated_frame = pose_results[0].plot()
        
        # 获取所有球员
        players = self.get_player_keypoints(pose_results)
        
        # 检测持球球员
        carrier_id = None
        carrier_confidence = 0
        carrier_keypoints = None
        
        if self.ball_detected and len(players) > 0:
            carrier_id, carrier_confidence = self.detect_ball_carrier(self.ball_position, players)
            
            # 获取持球球员的关键点
            if carrier_id is not None:
                for player in players:
                    if player['id'] == carrier_id:
                        carrier_keypoints = player['keypoints']
                        break
        
        # 追踪持球状态
        is_holding = self.track_ball_carrier(carrier_id, carrier_confidence)
        
        # 如果确认持球，进行运球和步数检测
        if is_holding and carrier_keypoints is not None:
            # 检测运球
            dribble_detected = self.detect_dribble()
            
            # 如果检测到运球，重置步数计数
            if dribble_detected:
                self.step_count = 0
                print(f"帧 {self.current_frame}: 运球后重置步数计数")
            else:
                # 统计持球球员的步数
                new_steps = self.count_carrier_steps(carrier_keypoints)
                if new_steps > 0:
                    self.step_count += new_steps
                    self.total_steps += new_steps
                    print(f"帧 {self.current_frame}: 持球球员 {self.ball_carrier_id} 步数+{new_steps}, 总步数: {self.step_count}")
        
        # 检测走步违例
        if is_holding:
            if self.detect_travel_violation():
                if not self.travel_violation:
                    self.travel_violation = True
                    self.travel_start_frame = self.current_frame
                    self.travel_count += 1
                    self.travel_frames.append(self.current_frame)
                    print(f"帧 {self.current_frame}: 走步违例检测! 步数: {self.step_count}, 运球: {self.dribble_count}")
                    
                    # 播放警告声音
                    try:
                        winsound.Beep(1000, 500)
                    except:
                        pass
            else:
                # 如果没有走步违例，确保状态重置
                if self.travel_violation and self.current_frame - self.travel_start_frame > 30:
                    self.travel_violation = False
        
        # 绘制检测结果和信息
        frame = self.draw_detection_info(frame, pose_annotated_frame, carrier_id, is_holding)
        
        return frame
    
    def draw_detection_info(self, frame, pose_frame, carrier_id, is_holding):
        """在帧上绘制检测信息"""
        # 合并篮球和姿态检测结果
        combined_frame = cv2.addWeighted(frame, 0.6, pose_frame, 0.4, 0)
        
        # 高亮显示持球球员
        if carrier_id is not None:
            # 在持球球员周围绘制彩色边界框
            cv2.putText(
                combined_frame, f"Ball Carrier: {carrier_id}", 
                (self.frame_width - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
            )
        
        # 绘制状态信息
        info_lines = [
            f"Frame: {self.current_frame}/{self.total_frames}",
            f"Ball: {'Detected' if self.ball_detected else 'Not Detected'}",
            f"Ball Carrier: {self.ball_carrier_id if self.ball_carrier_id is not None else 'None'}",
            f"Holding: {'Yes' if is_holding else 'No'} ({self.holding_frames} frames)",
            f"Dribble Count: {self.dribble_count} (Total: {self.total_dribbles})",
            f"Step Count: {self.step_count} (Total: {self.total_steps})",
            f"Max Steps Allowed: 2 (without dribble)",
            f"Travel Violation: {'DETECTED!' if self.travel_violation else 'No'}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 30
            color = (0, 0, 255) if "VIOLATION" in line else (0, 255, 0) if "DETECTED" in line else (255, 255, 255)
            thickness = 2 if "VIOLATION" in line or "DETECTED" in line else 1
            
            cv2.putText(
                combined_frame, line, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness + 1, cv2.LINE_AA
            )
            cv2.putText(
                combined_frame, line, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness, cv2.LINE_AA
            )
        
        # 如果检测到走步，添加视觉提示
        if self.travel_violation:
            # 红色半透明覆盖
            overlay = combined_frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 255), -1)
            combined_frame = cv2.addWeighted(overlay, 0.3, combined_frame, 0.7, 0)
            
            # 走步警告文本
            warning_text = "TRAVEL VIOLATION!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
            text_x = (self.frame_width - text_size[0]) // 2
            text_y = (self.frame_height + text_size[1]) // 2
            
            cv2.putText(
                combined_frame, warning_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA
            )
            
            # 显示走步计数
            count_text = f"Violations: {self.travel_count}"
            cv2.putText(
                combined_frame, count_text, (text_x, text_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )
        
        # 显示进度条
        if self.total_frames > 0:
            progress = self.current_frame / self.total_frames
            bar_width = int(self.frame_width * 0.8)
            bar_height = 20
            bar_x = (self.frame_width - bar_width) // 2
            bar_y = self.frame_height - 40
            
            # 进度条背景
            cv2.rectangle(combined_frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # 进度条前景
            progress_width = int(bar_width * progress)
            cv2.rectangle(combined_frame, (bar_x, bar_y), 
                         (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
            
            # 进度文本
            progress_text = f"{progress * 100:.1f}%"
            text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = bar_x + (bar_width - text_size[0]) // 2
            text_y = bar_y + (bar_height + text_size[1]) // 2
            
            cv2.putText(
                combined_frame, progress_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
        
        return combined_frame
    
    def save_analysis_report(self):
        """保存分析报告"""
        report = {
            "video_file": self.video_path,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_info": {
                "width": self.frame_width,
                "height": self.frame_height,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "duration_seconds": self.total_frames / self.fps if self.fps > 0 else 0
            },
            "detection_results": {
                "total_dribbles": self.total_dribbles,
                "total_steps": self.total_steps,
                "travel_violations": self.travel_count,
                "travel_frames": self.travel_frames,
                "final_judgment": "Travel Violation Detected" if self.travel_count > 0 else "No Travel Violation"
            },
            "analysis_notes": {
                "ball_detected": self.ball_detected,
                "carrier_tracked": self.ball_carrier_id is not None,
                "step_count_explanation": "Steps counted only for ball carrier when holding ball",
                "dribble_count_explanation": "Dribbles counted based on vertical ball motion"
            }
        }
        
        # 保存JSON报告
        report_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存文本报告
        text_report_path = os.path.join(self.output_dir, "analysis_report.txt")
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("篮球走步违例分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"视频文件: {report['video_file']}\n")
            f.write(f"分析时间: {report['analysis_date']}\n\n")
            
            f.write("视频信息:\n")
            f.write(f"  分辨率: {report['video_info']['width']}x{report['video_info']['height']}\n")
            f.write(f"  帧率: {report['video_info']['fps']:.2f} FPS\n")
            f.write(f"  总帧数: {report['video_info']['total_frames']}\n")
            f.write(f"  时长: {report['video_info']['duration_seconds']:.2f}秒\n\n")
            
            f.write("检测结果:\n")
            f.write(f"  总运球次数: {report['detection_results']['total_dribbles']}\n")
            f.write(f"  总步数: {report['detection_results']['total_steps']}\n")
            f.write(f"  走步违例次数: {report['detection_results']['travel_violations']}\n")
            
            if report['detection_results']['travel_frames']:
                f.write(f"  走步发生帧: {report['detection_results']['travel_frames']}\n")
            
            f.write(f"  最终判定: {report['detection_results']['final_judgment']}\n\n")
            
            f.write("分析说明:\n")
            f.write(f"  篮球检测: {'成功' if report['analysis_notes']['ball_detected'] else '失败'}\n")
            f.write(f"  持球球员追踪: {'成功' if report['analysis_notes']['carrier_tracked'] else '失败'}\n")
            f.write(f"  步数统计: {report['analysis_notes']['step_count_explanation']}\n")
            f.write(f"  运球统计: {report['analysis_notes']['dribble_count_explanation']}\n")
            f.write("=" * 60 + "\n")
        
        print(f"分析报告已保存: {report_path}")
        print(f"文本报告已保存: {text_report_path}")
        
        return report
    
    def analyze_video(self, show_preview=True):
        """分析整个视频"""
        print("开始分析视频...")
        
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            # 处理当前帧
            processed_frame = self.process_frame(frame)
            
            # 写入输出视频
            self.video_writer.write(processed_frame)
            
            # 显示预览
            if show_preview:
                cv2.imshow("Basketball Travel Analysis", processed_frame)
                
                # 按'q'退出，按's'暂停/继续
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户中断分析")
                    break
                elif key == ord('s'):
                    print("暂停，按任意键继续...")
                    cv2.waitKey(0)
                    print("继续分析...")
            
            # 显示进度
            if self.current_frame % 30 == 0 and self.total_frames > 0:
                elapsed_time = time.time() - self.start_time
                speed = self.current_frame / elapsed_time if elapsed_time > 0 else 0
                remaining_frames = self.total_frames - self.current_frame
                eta = remaining_frames / speed if speed > 0 else 0
                
                progress = (self.current_frame / self.total_frames) * 100
                print(f"进度: {self.current_frame}/{self.total_frames} ({progress:.1f}%) | "
                      f"速度: {speed:.1f} FPS | ETA: {eta:.1f}秒 | "
                      f"违例: {self.travel_count}")
        
        # 完成分析
        self.finish_analysis()
    
    def finish_analysis(self):
        """完成分析"""
        total_time = time.time() - self.start_time
        
        # 释放资源
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        
        # 保存分析报告
        report = self.save_analysis_report()
        
        # 打印最终结果
        print("\n" + "=" * 60)
        print("视频分析完成!")
        print("=" * 60)
        
        print(f"输入视频: {self.video_path}")
        print(f"输出视频: {self.output_video_path}")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"平均处理速度: {self.current_frame/total_time:.2f} FPS\n")
        
        print("检测统计:")
        print(f"  总运球次数: {self.total_dribbles}")
        print(f"  总步数: {self.total_steps} (仅统计持球球员)")
        print(f"  走步违例次数: {self.travel_count}")
        
        if self.travel_frames:
            print(f"  走步发生帧: {self.travel_frames}")
        
        print(f"\n最终判定: {report['detection_results']['final_judgment']}")
        print("=" * 60)
        
        return report

def main():
    # 输入视频文件路径
    video_path = r"C:\Users\李国靖\Desktop\AI-Basketball-Referee-main2\AI-Basketball-Referee-main\QQ2025122-222737.mp4"
    
    # 如果文件不存在，提示用户输入
    if not os.path.exists(video_path):
        print("默认视频文件不存在，请手动输入视频路径")
        video_path = input("请输入视频文件路径: ").strip().strip('"')
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    try:
        # 创建分析器并运行
        analyzer = BasketballTravelAnalyzer(video_path)
        analyzer.analyze_video(show_preview=True)
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()