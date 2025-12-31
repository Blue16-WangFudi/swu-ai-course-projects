import threading
import queue
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from video_offside_cli import process_video


class VideoOffsideApp:
    """
    简洁视频越位检测UI：
    - 选择视频、设置帧间隔/队伍编号
    - 实时预览关键帧和进度
    - 结束后给出越位摘要
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("越位检测 - 精简版")
        self.root.geometry("900x700")

        self.video_path = tk.StringVar()
        self.off_team = tk.IntVar(value=0)
        self.def_team = tk.IntVar(value=1)
        self.frame_stride = tk.IntVar(value=5)
        self.preview_stride = tk.IntVar(value=15)

        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="请选择视频并开始检测")
        self.debug_var = tk.StringVar(value="")

        self.result_queue: queue.Queue = queue.Queue(maxsize=30)
        self.preview_photo = None
        self.running = False
        self.extended_photo = None  # 新增：用于存储连线图的PhotoImage
        self.first_offside_frame = None  # 新增：记录第一次越位帧

        # ========== 创建可滚动容器 ==========
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建画布和滚动条
        self.scroll_canvas = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.scroll_canvas)

        # 配置滚动区域
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
        )

        # 将可滚动框架放在画布上
        self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # 支持 Windows 和 Mac
        self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # 支持 Linux
        self.scroll_canvas.bind_all("<Button-4>", lambda e: self.scroll_canvas.yview_scroll(-1, "units"))
        self.scroll_canvas.bind_all("<Button-5>", lambda e: self.scroll_canvas.yview_scroll(1, "units"))

        # 调整画布大小时，更新内部框架宽度
        def configure_canvas(event):
            self.scroll_canvas.itemconfig(self.scroll_canvas_window, width=event.width)

        self.scroll_canvas.bind("<Configure>", configure_canvas)

        # 打包画布和滚动条
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 重新设置画布窗口的引用
        self.scroll_canvas_window = self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        top = ttk.Frame(self.scrollable_frame, padding=10)  # 使用 scrollable_frame
        top.pack(fill=tk.X)

        ttk.Label(top, text="视频路径:").pack(side=tk.LEFT)
        path_entry = ttk.Entry(top, textvariable=self.video_path, width=60)
        path_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="选择文件", command=self._choose_file).pack(side=tk.LEFT)

        params = ttk.Frame(self.scrollable_frame, padding=10)  # 使用 scrollable_frame
        params.pack(fill=tk.X)
        ttk.Label(params, text="进攻队编号").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.off_team, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(params, text="防守队编号").grid(row=0, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.def_team, width=6).grid(row=0, column=3, padx=5)
        ttk.Label(params, text="帧采样间隔").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Spinbox(params, from_=1, to=10, textvariable=self.frame_stride, width=6).grid(row=1, column=1, padx=5)
        ttk.Label(params, text="预览间隔").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(params, from_=1, to=30, textvariable=self.preview_stride, width=6).grid(row=1, column=3, padx=5)
        ttk.Button(params, text="开始检测", command=self.start_detection).grid(row=0, column=4, rowspan=2, padx=10)

        legend_frame = ttk.Frame(self.scrollable_frame, padding=(10, 0))  # 使用 scrollable_frame
        legend_frame.pack(fill=tk.X)
        legend_text = "💡 提示：预览中蓝色边界框 = Team 0，红色边界框 = Team 1。每个球员边界框上方会显示 'Team 0' 或 'Team 1' 标签。"
        legend_label = ttk.Label(legend_frame, text=legend_text, foreground="#666", font=("Arial", 9))
        legend_label.pack(side=tk.LEFT, padx=10)

        # 修改canvas_frame部分，创建左右两个canvas
        canvas_frame = ttk.LabelFrame(self.scrollable_frame, text="预览", padding=10)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建左右并排的容器
        preview_container = ttk.Frame(canvas_frame)
        preview_container.pack(fill=tk.BOTH, expand=True)

        # 左半部分：实时预览
        left_frame = ttk.Frame(preview_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(left_frame, text="实时检测预览", anchor="center").pack()
        self.preview_canvas = tk.Canvas(left_frame, width=400, height=450, bg="#111")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self._show_preview_canvas_text("等待检测...")

        # 右半部分：越位连线图
        right_frame = ttk.Frame(preview_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(right_frame, text="首次越位连线图", anchor="center").pack()
        self.extended_canvas = tk.Canvas(right_frame, width=150, height=100, bg="#111")
        self.extended_canvas.pack(fill=tk.BOTH, expand=True)
        self._show_extended_canvas_text("未检测到越位")

        status_frame = ttk.Frame(self.scrollable_frame, padding=10)  # 使用 scrollable_frame
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Progressbar(status_frame, variable=self.progress_var, length=200, mode="determinate").pack(side=tk.RIGHT)

        summary_frame = ttk.LabelFrame(self.scrollable_frame, text="检测摘要", padding=10)  # 使用 scrollable_frame
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 创建滚动条和文本区域
        scroll_frame = ttk.Frame(summary_frame)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        summary_scrollbar = ttk.Scrollbar(scroll_frame)  # 重命名避免冲突
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.summary_text_widget = tk.Text(
            scroll_frame,
            wrap=tk.WORD,
            yscrollcommand=summary_scrollbar.set,
            height=6,
            font=("Arial", 9),
            bg="#f5f5f5",
            fg="#333"
        )
        self.summary_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scrollbar.config(command=self.summary_text_widget.yview)

        self.debug_label = ttk.Label(summary_frame, textvariable=self.debug_var, anchor="w", justify=tk.LEFT,
                                     foreground="#888")
        self.debug_label.pack(fill=tk.X, pady=(5, 0))

    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if path:
            self.video_path.set(path)

    def start_detection(self):
        # 重置状态
        self.first_offside_frame = None
        self._show_extended_canvas_text("未检测到越位")
        if self.running:
            messagebox.showinfo("提示", "检测正在进行，请稍候。")
            return
        video = self.video_path.get()
        if not video or not Path(video).exists():
            messagebox.showerror("错误", "请先选择有效的视频文件")
            return

        try:
            off = int(self.off_team.get())
            deff = int(self.def_team.get())
        except Exception:
            messagebox.showerror("错误", "请输入有效的队伍编号（整数）")
            return

        self.status_var.set("开始处理视频...")
        self.summary_text_widget.delete(1.0, tk.END)
        self.summary_text_widget.insert(1.0, "正在处理视频，请稍候...")
        self.debug_var.set("")
        self.progress_var.set(0)
        self.running = True
        self._show_canvas_text("读取视频中...")

        thread = threading.Thread(
            target=self._run_detection,
            args=(video, off, deff, self.frame_stride.get(), self.preview_stride.get()),
            daemon=True,
        )
        thread.start()

    def _run_detection(self, video, off, deff, stride, preview_stride):
        def on_progress(frame_id, total):
            self.result_queue.put(("progress", (frame_id, total)))

        def on_frame(frame_id, frame, is_offside):
            self.result_queue.put(("frame", (frame_id, frame, is_offside)))

        # 新增：越位连线图回调
        def on_offside_extended(frame_id, extended_img):
            self.result_queue.put(("offside_extended", (frame_id, extended_img)))

        try:
            stats = process_video(
                video_path=video,
                off_team=off,
                def_team=deff,
                frame_stride=max(1, stride),
                preview_stride=max(1, preview_stride),
                on_progress=on_progress,
                on_frame=on_frame,
                on_offside_extended=on_offside_extended,
            )
            self.result_queue.put(("done", stats))
        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg, payload = self.result_queue.get_nowait()
                if msg == "progress":
                    frame_id, total = payload
                    self._update_progress(frame_id, total)
                elif msg == "frame":
                    frame_id, frame, is_offside = payload
                    self._update_frame(frame_id, frame, is_offside)
                elif msg == "offside_extended":  # 新增：处理连线图消息
                    frame_id, extended_img = payload
                    self._update_extended_img(frame_id, extended_img)
                elif msg == "done":
                    self._on_finish(payload)
                elif msg == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

        # 新增：更新连线图的方法

    def _update_extended_img(self, frame_id, extended_img):
        if extended_img is None or extended_img.size == 0:
            self.debug_var.set(f"帧 {frame_id} 连线图数据为空")
            return

        # 如果是第一次检测到越位
        if self.first_offside_frame is None:
            self.first_offside_frame = frame_id

            # 转换图像格式
            rgb = cv2.cvtColor(extended_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # 在Canvas上显示缩略图
            img.thumbnail((400, 300))
            self.extended_photo = ImageTk.PhotoImage(img)

            self.extended_canvas.delete("all")
            self.extended_canvas.create_image(200, 150, image=self.extended_photo)
            self.extended_canvas.create_text(200, 20, text=f"首次越位帧: {frame_id}",
                                             fill="white", font=("Arial", 12, "bold"))

            # 添加"点击查看大图"按钮
            view_btn = tk.Button(
                self.extended_canvas,
                text="点击查看大图",
                command=lambda: self._show_large_extended_img(frame_id, extended_img),
                bg="blue", fg="white", font=("Arial", 10)
            )
            self.extended_canvas.create_window(200, 280, window=view_btn)

    # 新增：显示连线图canvas的文本
    def _show_extended_canvas_text(self, text):
        self.extended_canvas.delete("all")
        self.extended_canvas.create_text(
            200, 225,
            text=text,
            fill="#999",
            font=("Arial", 14)
        )

    def _show_large_extended_img(self, frame_id, img_array):
        """在新窗口中显示大图"""
        top = tk.Toplevel(self.root)
        top.title(f"越位连线图 - 帧 {frame_id}")
        top.geometry("1000x800")

        # 转换图像
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # 调整尺寸以适应窗口
        img_width, img_height = img.size
        scale = min(900 / img_width, 700 / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img)

        # 显示图片
        label = tk.Label(top, image=photo)
        label.image = photo  # 保持引用
        label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 添加说明
        info = tk.Label(top, text="球员位置图",
                        font=("Arial", 12))
        info.pack(pady=(0, 10))

    def _update_progress(self, frame_id, total):
        progress = min(100, (frame_id / total) * 100) if total else 0
        self.progress_var.set(progress)
        self.status_var.set(f"处理中... {progress:.1f}%")

    def _update_frame(self, frame_id, frame, is_offside):
        if frame is None or frame.size == 0:
            self.debug_var.set(f"帧 {frame_id} 空数据，跳过")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((820, 460))
        self.preview_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")  # 改为 preview_canvas
        self.preview_canvas.create_image(400, 230, image=self.preview_photo)  # 改为 preview_canvas
        flag = "🚩 越位" if is_offside else "✅ 正常"
        self.status_var.set(f"帧 {frame_id} | {flag}")
        self.debug_var.set("")

    def _on_finish(self, stats):
        self.running = False
        events = stats.get("unique_events", [])
        total_frames = stats.get("total_frames", 0)
        fps = stats.get("fps", 0)

        # 如果没有检测到越位，更新连线图区域
        if not events and self.first_offside_frame is None:
            self._show_extended_canvas_text("未检测到越位")

        self.summary_text_widget.delete(1.0, tk.END)
        if events:
            summary = f"检测到 {len(events)} 次疑似越位事件\n"
            summary += f"起始帧号: {', '.join(map(str, events))}\n"
            summary += f"总越位帧数: {len(stats.get('offside_frames', []))}\n"
            summary += f"视频总帧数: {total_frames} | FPS: {fps:.2f}\n"
            summary += f"结果目录: {stats.get('output_dir')}\n\n"
            summary += "注意：检测结果仅供参考，可能存在误判。请结合视频内容人工复核。"
        else:
            summary = f"未检测到越位事件\n"
            summary += f"视频总帧数: {total_frames} | FPS: {fps:.2f}\n"
            summary += f"结果目录: {stats.get('output_dir')}"

        self.summary_text_widget.insert(1.0, summary)
        self.status_var.set("处理完成")
        self.progress_var.set(100)

    def _on_error(self, msg):
        self.running = False
        self.status_var.set("处理失败")
        self._show_canvas_text("处理失败")
        messagebox.showerror("错误", msg)

    def _show_canvas_text(self, text):
        self.preview_canvas.delete("all")  # 改为 preview_canvas
        self.preview_canvas.create_text(400, 230, text=text, fill="#999", font=("Arial", 16))  # 改为 preview_canvas

    # 添加一个别名方法以兼容现有代码
    def _show_preview_canvas_text(self, text):
        self._show_canvas_text(text)


def main():
    root = tk.Tk()
    app = VideoOffsideApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
