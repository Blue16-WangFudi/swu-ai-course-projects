import cv2
import numpy as np
import time
import argparse
import os
import traceback
from datetime import datetime
from ultralytics import YOLO


class DoubleDribbleDetector:
    def __init__(self, video_path=None, output_video=None, log_file=None, show_window=False):
        """
        初始化双带违规检测器
        
        Args:
            video_path: 输入视频文件路径（None则使用摄像头）
            output_video: 输出视频文件路径（None则不保存）
            log_file: 日志文件路径（None则不保存日志）
            show_window: 是否显示视频窗口
        """
        # 加载YOLO模型
        print("正在加载YOLO模型...")
        try:
            self.pose_model = YOLO("yolov8s-pose.pt")
            print("姿态模型加载完成！")
        except Exception as e:
            print(f"加载姿态模型失败: {e}")
            print("请确保 yolov8s-pose.pt 文件在当前目录")
            raise
        
        try:
            self.ball_model = YOLO("basketballModel.pt")
            print("篮球检测模型加载完成！")
        except Exception as e:
            print(f"加载篮球检测模型失败: {e}")
            print("注意: 篮球检测模型可能需要单独训练")
            print("尝试使用通用模型...")
            try:
                self.ball_model = YOLO("yolov8n.pt")
                print("使用通用检测模型代替篮球检测")
            except:
                self.ball_model = None
                print("警告: 无法加载任何检测模型")

        # 检查视频文件是否存在
        if video_path:
            if not os.path.exists(video_path):
                print(f"错误: 视频文件不存在: {video_path}")
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            self.video_source = "file"
            print(f"正在读取视频文件: {video_path}")
        else:
            raise ValueError("请提供视频文件路径")

        # 获取原始视频属性
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            self.fps = 30
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        
        print(f"原始分辨率: {self.original_width}x{self.original_height}")
        print(f"帧率: {self.fps} FPS")
        print(f"总帧数: {self.total_frames}")
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        print(f"视频时长: {duration:.2f}秒")

        # 设置目标分辨率（调整视频尺寸）
        # 保持原始比例，但限制最大尺寸
        max_width = 1280
        max_height = 720
        
        # 计算缩放比例
        scale = min(max_width / self.original_width, max_height / self.original_height)
        
        self.frame_width = int(self.original_width * scale)
        self.frame_height = int(self.original_height * scale)
        
        print(f"处理分辨率: {self.frame_width}x{self.frame_height}")

        # 初始化视频写入器（如果需要保存）
        self.output_writer = None
        self.show_window = show_window
        if output_video:
            if not output_video.endswith('.mp4'):
                output_video = output_video + '.mp4'
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_writer = cv2.VideoWriter(
                output_video, fourcc, self.fps, 
                (self.frame_width, self.frame_height)
            )
            print(f"输出视频将保存到: {output_video}")

        # 人体关键点索引
        self.body_index = {"left_wrist": 10, "right_wrist": 9}

        # 状态变量
        self.hold_start_time = None
        self.is_holding = False
        self.was_holding = False
        self.last_dribble_time = None

        # 参数设置 - 针对缩放后的视频调整
        self.hold_duration = 0.2  # 持球判定持续时间(秒) - 减小
        self.hold_threshold = 100  # 持球距离阈值(像素) - 减小
        self.dribble_threshold = 50  # 运球检测阈值 - 增加
        self.min_dribble_interval = 0.2  # 最小运球间隔(秒)

        # 跟踪变量
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.last_ball_position = None

        # 检测结果
        self.double_dribble_detected = False  # 是否发生双带违规
        self.double_dribble_time = None
        self.double_dribble_count = 0
        self.double_dribble_frames = []
        
        # 性能统计
        self.start_time = time.time()
        self.processed_frames = 0
        self.detection_stats = {
            "human_detected": 0,
            "ball_detected": 0,
            "holding_frames": 0,
            "dribble_events": 0
        }
        
        # 调试信息存储
        self.debug_info = []

    def resize_frame(self, frame):
        """
        调整帧大小并保持宽高比
        
        Args:
            frame: 原始帧
            
        Returns:
            调整大小后的帧
        """
        return cv2.resize(frame, (self.frame_width, self.frame_height))

    def run(self):
        """
        运行主检测循环
        """
        print("\n开始检测双带违规...")
        print("正在处理视频，请稍候...\n")
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("视频处理完成！")
                    break
                
                self.frame_count += 1
                self.processed_frames += 1
                
                # 调整帧大小
                frame_resized = self.resize_frame(frame)
                
                # 处理帧并检测
                processed_frame, had_double_dribble = self.process_frame(frame_resized)
                
                # 如果检测到双带违规
                if had_double_dribble:
                    self.double_dribble_detected = True
                
                # 保存到输出视频（如果需要）
                if self.output_writer is not None:
                    self.output_writer.write(processed_frame)
                
                # 显示视频窗口（如果需要）
                if self.show_window:
                    cv2.imshow("双带违规检测", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 显示进度
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    print(f"处理进度: {progress:.1f}% ({self.frame_count}/{self.total_frames} 帧)", end='\r')
        
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        
        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")
            traceback.print_exc()
        
        finally:
            # 释放资源
            self.cap.release()
            if self.output_writer is not None:
                self.output_writer.release()
            if self.show_window:
                cv2.destroyAllWindows()
            
            # 打印最终结果
            self.print_final_result()

    def process_frame(self, frame):
        """
        处理单个视频帧
        
        Args:
            frame: 输入帧
            
        Returns:
            processed_frame: 处理后的帧
            had_double_dribble: 本帧是否检测到双带违规
        """
        # 复制原始帧用于显示
        display_frame = frame.copy()
        
        # 姿态检测
        had_double_dribble = False
        left_wrist = None
        right_wrist = None
        
        try:
            pose_results = self.pose_model(frame, verbose=False, conf=0.3)
            
            # 获取关键点
            keypoints = pose_results[0].keypoints
            
            if keypoints is not None and len(keypoints) > 0:
                try:
                    # 将keypoints移到CPU并转换为numpy
                    if hasattr(keypoints, 'xy'):
                        keypoints_np = keypoints.xy.cpu().numpy()
                    else:
                        keypoints_np = keypoints.cpu().numpy()
                    
                    if len(keypoints_np) > 0 and len(keypoints_np[0]) > self.body_index["left_wrist"]:
                        left_wrist = keypoints_np[0][self.body_index["left_wrist"]]
                        right_wrist = keypoints_np[0][self.body_index["right_wrist"]]
                        self.detection_stats["human_detected"] += 1
                        
                        # 绘制手腕点
                        cv2.circle(display_frame, (int(left_wrist[0]), int(left_wrist[1])), 6, (255, 0, 0), -1)
                        cv2.circle(display_frame, (int(right_wrist[0]), int(right_wrist[1])), 6, (0, 255, 0), -1)
                except Exception as e:
                    pass
        except Exception as e:
            pass
        
        # 篮球检测
        ball_detected = False
        ball_x_center = None
        ball_y_center = None
        
        if self.ball_model is not None:
            try:
                ball_results_list = self.ball_model(frame, verbose=False, conf=0.4)
                
                for ball_results in ball_results_list:
                    if ball_results.boxes is not None and len(ball_results.boxes) > 0:
                        # 将boxes数据移到CPU
                        boxes_xyxy = ball_results.boxes.xyxy.cpu().numpy()
                        
                        for i, bbox in enumerate(boxes_xyxy):
                            # 获取边界框坐标
                            x1, y1, x2, y2 = bbox[:4]
                            
                            # 计算篮球中心
                            ball_x_center = (x1 + x2) / 2
                            ball_y_center = (y1 + y2) / 2
                            
                            # 保存当前篮球位置
                            current_ball_position = (ball_x_center, ball_y_center)
                            
                            # 更新运球计数
                            self.update_dribble_count(ball_x_center, ball_y_center)
                            
                            # 保存当前中心点供下一帧使用
                            self.prev_x_center = ball_x_center
                            self.prev_y_center = ball_y_center
                            
                            ball_detected = True
                            self.detection_stats["ball_detected"] += 1
                            
                            # 如果检测到手腕，计算距离
                            if left_wrist is not None and right_wrist is not None:
                                left_distance = np.hypot(
                                    ball_x_center - left_wrist[0], ball_y_center - left_wrist[1]
                                )
                                right_distance = np.hypot(
                                    ball_x_center - right_wrist[0], ball_y_center - right_wrist[1]
                                )
                                
                                # 检查是否持球
                                self.check_holding(left_distance, right_distance)
                            
                            # 绘制篮球边界框
                            cv2.rectangle(
                                display_frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 0),
                                2,
                            )
                            
                            # 绘制篮球中心点
                            cv2.circle(
                                display_frame,
                                (int(ball_x_center), int(ball_y_center)),
                                5,
                                (0, 255, 0),
                                -1,
                            )
                            
                            # 保存篮球位置
                            self.last_ball_position = current_ball_position
            except Exception as e:
                pass
        
        # 如果没有检测到篮球，重置持球状态
        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False
        elif self.is_holding:
            self.detection_stats["holding_frames"] += 1
        
        # 检查是否发生双带违规
        had_double_dribble = self.check_double_dribble()
        
        # 如果检测到双带违规，添加红色叠加层
        if had_double_dribble:
            red_tint = np.full_like(display_frame, (0, 0, 255), dtype=np.uint8)
            display_frame = cv2.addWeighted(display_frame, 0.7, red_tint, 0.3, 0)
            
            # 添加文字提示
            cv2.putText(
                display_frame,
                "DOUBLE DRIBBLE!",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
        
        # 如果正在持球，添加蓝色叠加层
        if self.is_holding:
            blue_tint = np.full_like(display_frame, (255, 0, 0), dtype=np.uint8)
            display_frame = cv2.addWeighted(display_frame, 0.8, blue_tint, 0.2, 0)
            
            # 添加文字提示
            cv2.putText(
                display_frame,
                "HOLDING BALL",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        
        # 添加状态信息
        cv2.putText(
            display_frame,
            f"Frame: {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            display_frame,
            f"Holding: {'Yes' if self.is_holding else 'No'}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if self.is_holding else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            display_frame,
            f"Dribble Count: {self.dribble_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        return display_frame, had_double_dribble

    def check_holding(self, left_distance, right_distance):
        """
        检查篮球是否被持球
        
        Args:
            left_distance: 左腕到篮球的距离
            right_distance: 右腕到篮球的距离
        """
        # 如果篮球靠近任一手腕
        min_distance = min(left_distance, right_distance)
        if min_distance < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif time.time() - self.hold_start_time > self.hold_duration:
                self.is_holding = True
                self.was_holding = True
                self.dribble_count = 0  # 重置运球计数
                
                # 记录调试信息
                if self.frame_count % 30 == 0:  # 每30帧记录一次
                    self.debug_info.append(f"帧 {self.frame_count}: 开始持球，距离: {min_distance:.1f}")
        else:
            if self.is_holding:
                self.debug_info.append(f"帧 {self.frame_count}: 结束持球")
            self.hold_start_time = None
            self.is_holding = False

    def update_dribble_count(self, x_center, y_center):
        """
        更新运球计数
        
        Args:
            x_center: 篮球x坐标
            y_center: 篮球y坐标
        """
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center
            
            # 检查是否有显著的向下运动（运球）
            if (self.prev_delta_y is not None and 
                delta_y < 0 and 
                self.prev_delta_y > self.dribble_threshold and
                (self.last_dribble_time is None or time.time() - self.last_dribble_time > self.min_dribble_interval)):
                
                self.dribble_count += 1
                self.last_dribble_time = time.time()
                self.detection_stats["dribble_events"] += 1
                
                # 记录调试信息
                if self.frame_count % 30 == 0:  # 每30帧记录一次
                    self.debug_info.append(f"帧 {self.frame_count}: 检测到运球，计数: {self.dribble_count}")
            
            self.prev_delta_y = delta_y

    def check_double_dribble(self):
        """
        检查是否发生双带违规
        
        Returns:
            bool: 是否检测到双带违规
        """
        # 双带违规的逻辑：持球后再次运球
        if self.was_holding and self.dribble_count > 0:
            # 记录双带违规
            self.double_dribble_time = time.time()
            self.double_dribble_count += 1
            
            # 计算时间戳
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_number = self.frame_count
            
            # 记录到列表
            self.double_dribble_frames.append({
                "frame": frame_number,
                "time": current_time,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # 记录调试信息
            self.debug_info.append(f"[双带违规 #{self.double_dribble_count}] 帧: {frame_number}, 时间: {current_time:.2f}秒")
            
            # 重置状态
            self.was_holding = False
            self.dribble_count = 0
            
            return True
        
        return False

    def print_final_result(self):
        """
        打印最终检测结果
        """
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        
        print("\n" + "="*60)
        print("双带违规检测结果")
        print("="*60)
        print(f"视频处理时长: {elapsed_time:.1f} 秒")
        print(f"处理帧数: {self.processed_frames}")
        print(f"处理速度: {self.processed_frames/elapsed_time:.1f} FPS")
        print(f"检测到人体: {self.detection_stats['human_detected']} 帧")
        print(f"检测到篮球: {self.detection_stats['ball_detected']} 帧")
        print(f"持球帧数: {self.detection_stats['holding_frames']}")
        print(f"运球事件: {self.detection_stats['dribble_events']} 次")
        
        print("\n" + "-"*60)
        
        if self.double_dribble_count > 0:
            print(f"🚨 检测到双带违规！总共发生 {self.double_dribble_count} 次")
            print("\n双带违规发生时间点:")
            for i, event in enumerate(self.double_dribble_frames, 1):
                print(f"  {i}. 第 {event['frame']} 帧 | 时间 {event['time']:.2f}秒")
            
            # 判断是否发生了真正的二运犯规（篮球规则）
            # 真正的双带违规是：持球后再次开始运球
            if self.detection_stats['holding_frames'] > 5 and self.double_dribble_count > 0:
                print(f"\n🏀 最终判定：视频中发生了双带违规（二运犯规）！")
            else:
                print(f"\n⚠️  检测到可能的双带违规，但持球时间较短，可能是误判")
        else:
            print("✅ 未检测到双带违规")
            print(f"\n🏀 最终判定：视频中未发生双带违规（二运犯规）")
        
        # 打印最近的调试信息
        if self.debug_info:
            print(f"\n调试信息（最近{min(10, len(self.debug_info))}条）:")
            for i, info in enumerate(self.debug_info[-10:], 1):
                print(f"  {info}")
        
        print("="*60)


def main():
    """
    主函数，处理命令行参数并运行检测器
    """
    parser = argparse.ArgumentParser(description='篮球双带违规检测系统')
    
    # 设置当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 检查当前目录下的视频文件
    video_files = []
    for file in os.listdir(current_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            video_files.append(file)
    
    default_video = None
    if video_files:
        print("\n找到以下视频文件:")
        for i, file in enumerate(video_files, 1):
            print(f"  {i}. {file}")
            if 'dribble' in file.lower() or 'duibble' in file.lower():
                default_video = file
        
        if default_video is None:
            default_video = video_files[0]
        
        print(f"\n将使用默认视频: {default_video}")
    else:
        print("\n当前目录未找到视频文件")
    
    parser.add_argument('--video', type=str, 
                       default=default_video,
                       help='输入视频文件路径')
    parser.add_argument('--output', type=str, default="output_detection.mp4",
                       help='输出视频文件路径（默认: output_detection.mp4）')
    parser.add_argument('--show', action='store_true',
                       help='显示视频窗口（默认不显示）')
    
    args = parser.parse_args()
    
    try:
        # 如果未指定视频，询问用户
        if args.video is None:
            if video_files:
                print("\n请选择:")
                for i, file in enumerate(video_files, 1):
                    print(f"{i}. {file}")
                
                try:
                    choice = input(f"\n请选择文件编号 (1-{len(video_files)}): ").strip()
                    if choice and choice.isdigit():
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(video_files):
                            args.video = video_files[choice_idx]
                        else:
                            args.video = video_files[0]
                    else:
                        args.video = video_files[0]
                except:
                    args.video = video_files[0]
            else:
                print("没有可用的视频文件")
                return 1
        
        # 创建检测器实例
        detector = DoubleDribbleDetector(
            video_path=args.video,
            output_video=args.output,
            show_window=args.show
        )
        
        # 运行检测
        detector.run()
        
        print(f"\n处理完成！")
        if args.output:
            print(f"检测结果视频已保存到: {args.output}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())