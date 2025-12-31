import sys
import time
import threading
import csv
from pathlib import Path
from datetime import datetime
import argparse

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import yaml
except Exception:
    yaml = None

try:
    import tkinter as tk
    from tkinter import filedialog as tk_filedialog
except Exception:
    tk = None
    tk_filedialog = None


class RealtimeGlassesApp:
    def __init__(self, config_path: Path | None = None):
        self.cap = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.last_frame_time = 0.0
        self.fps = 0.0
        self.fps_smooth = 0.0
        self.acc_frames = 0
        self.acc_correct = 0
        self.acc_enabled = False
        self.gt_label = None
        self.last_frame = None
        self.named = False

        self.model = None
        self.face_detector = None

        self.cfg = {
            "model_path": str(Path("d:/code/glasses/runs/glasses_worldv2_full2/weights/best.pt")),
            "frame_width": 960,
            "yolo_imgsz": 640,
            "yolo_conf": 0.35,
            "yolo_iou": 0.5,
            "process_stride": 2,
            "max_faces": 3,
            "roi_expand": 0.2,
            "clahe": True,
            "log_path": str(Path("d:/code/glasses/logs/realtime.csv")),
            "output_dir": str(Path("d:/code/glasses/apps")),
        }

        if config_path and yaml is not None and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                self.cfg.update(loaded)

        self._init_model()
        self._init_face_detector()
        Path(self.cfg["log_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        self.log_fp = open(self.cfg["log_path"], "a", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_fp)
        if self.log_fp.tell() == 0:
            self.log_writer.writerow([
                "timestamp", "fps", "face_count", "glasses", "confidence", "bbox_xyxy"
            ])

    def _infer_single_frame(self, frame: np.ndarray):
        """Run one-shot inference on a frame (no FPS smoothing/logging)."""
        h, w = frame.shape[:2]
        target_w = self.cfg["frame_width"]
        if w != target_w:
            scale = target_w / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.cfg.get("clahe", True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        if self.face_detector is not None:
            try:
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
            except Exception as e:
                print(f"[WARN] detectMultiScale 失败，改用全图推理: {e}")
                faces = []
        else:
            faces = []
        faces = faces[: self.cfg["max_faces"]] if len(faces) > 0 else []

        wear = False
        conf = 0.0
        boxes_draw = []

        if len(faces) > 0:
            for (x, y, fw, fh) in faces:
                ex = int(fw * self.cfg["roi_expand"])
                ey = int(fh * self.cfg["roi_expand"])
                x0 = max(0, x - ex)
                y0 = max(0, y - ey)
                x1 = min(w, x + fw + ex)
                y1 = min(h, y + fh + ey)
                roi = frame[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                res = self.model.predict(
                    roi,
                    imgsz=self.cfg["yolo_imgsz"],
                    conf=self.cfg["yolo_conf"],
                    iou=self.cfg["yolo_iou"],
                    verbose=False,
                    device="cpu",
                )
                r0 = res[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    b = r0.boxes
                    idx = int(np.argmax(b.conf.cpu().numpy()))
                    bb = b.xyxy[idx].cpu().numpy().tolist()
                    c = float(b.conf[idx].cpu().numpy())
                    wear = True
                    conf = max(conf, c)
                    bx0 = int(bb[0] + x0)
                    by0 = int(bb[1] + y0)
                    bx1 = int(bb[2] + x0)
                    by1 = int(bb[3] + y0)
                    boxes_draw.append((bx0, by0, bx1, by1))

        color = (0, 200, 0) if wear else (0, 0, 255)
        label = "Glasses: YES" if wear else "Glasses: NO"

        for (x, y, x2, y2) in boxes_draw:
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Conf: {conf:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2, cv2.LINE_AA)
        return frame, wear, conf, boxes_draw

    def _init_model(self):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available. Please install 'ultralytics'.")
        model_path = self.cfg["model_path"]
        self.model = YOLO(model_path)

    def _init_face_detector(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            print(f"[WARN] Haar cascade 加载失败: {cascade_path}，将跳过人脸裁剪，直接全图推理。")
            self.face_detector = None

    def _open_camera(self, idx: int):
        backends = [None, getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None)]
        for be in backends:
            try:
                cap = cv2.VideoCapture(idx, be) if be is not None else cv2.VideoCapture(idx)
                if not cap.isOpened():
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["frame_width"])
                cap.set(cv2.CAP_PROP_FPS, 30)
                # probe a few frames
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    continue
                return cap
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
                continue
        return None

    def start(self, camera_index: int = 0):
        self.cap = self._open_camera(camera_index)
        if self.cap is None:
            print("[ERROR] 无法打开摄像头，请尝试更换索引或关闭占用程序。")
            return
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        self._ui_loop()

    def stop(self):
        self.running = False
        time.sleep(0.05)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        try:
            self.log_fp.flush()
            self.log_fp.close()
        except Exception:
            pass

    def _loop(self):
        frame_idx = 0
        last_pred = None
        last_conf = 0.0
        last_boxes = []
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = now - self.last_frame_time if self.last_frame_time else 0.0
            self.last_frame_time = now
            if dt > 0:
                self.fps = 1.0 / dt
                self.fps_smooth = 0.9 * self.fps_smooth + 0.1 * self.fps if self.fps_smooth else self.fps

            h, w = frame.shape[:2]
            target_w = self.cfg["frame_width"]
            if w != target_w:
                scale = target_w / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                h, w = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.cfg.get("clahe", True):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            if self.face_detector is not None:
                try:
                    faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                except Exception as e:
                    print(f"[WARN] detectMultiScale 失败，改用全图推理: {e}")
                    faces = []
            else:
                faces = []
            faces = faces[: self.cfg["max_faces"]] if len(faces) > 0 else []

            do_infer = (frame_idx % max(1, int(self.cfg["process_stride"])) == 0)
            wear = False
            conf = 0.0
            boxes_draw = []

            if len(faces) > 0 and do_infer:
                for (x, y, fw, fh) in faces:
                    ex = int(fw * self.cfg["roi_expand"])  # expand ROI horizontally
                    ey = int(fh * self.cfg["roi_expand"])  # expand ROI vertically
                    x0 = max(0, x - ex)
                    y0 = max(0, y - ey)
                    x1 = min(w, x + fw + ex)
                    y1 = min(h, y + fh + ey)
                    roi = frame[y0:y1, x0:x1]
                    if roi.size == 0:
                        continue
                    res = self.model.predict(
                        roi,
                        imgsz=self.cfg["yolo_imgsz"],
                        conf=self.cfg["yolo_conf"],
                        iou=self.cfg["yolo_iou"],
                        verbose=False,
                        device="cpu",
                    )
                    r0 = res[0]
                    if r0.boxes is not None and len(r0.boxes) > 0:
                        b = r0.boxes
                        idx = int(np.argmax(b.conf.cpu().numpy()))
                        bb = b.xyxy[idx].cpu().numpy().tolist()
                        c = float(b.conf[idx].cpu().numpy())
                        wear = True
                        conf = max(conf, c)
                        # translate ROI box to full-frame coords
                        bx0 = int(bb[0] + x0)
                        by0 = int(bb[1] + y0)
                        bx1 = int(bb[2] + x0)
                        by1 = int(bb[3] + y0)
                        boxes_draw.append((bx0, by0, bx1, by1))

                last_pred = wear
                last_conf = conf
                last_boxes = boxes_draw
            else:
                wear = last_pred if last_pred is not None else False
                conf = last_conf
                boxes_draw = last_boxes

            color = (0, 200, 0) if wear else (0, 0, 255)
            label = "Glasses: YES" if wear else "Glasses: NO"

            for (x, y, x2, y2) in boxes_draw:
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            for (x, y, fw, fh) in faces:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {self.fps_smooth:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2, cv2.LINE_AA)

            if self.acc_enabled and self.gt_label is not None:
                self.acc_frames += 1
                pred_yes = 1 if wear else 0
                gt_yes = 1 if self.gt_label else 0
                if pred_yes == gt_yes:
                    self.acc_correct += 1
                acc = (self.acc_correct / max(1, self.acc_frames)) * 100.0
                cv2.putText(frame, f"ACC: {acc:.1f}% (GT mode)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"DetectRate: { (1.0 if wear else 0.0) * 100:.0f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)

            ts = datetime.now().isoformat()
            bbox_str = "|".join([f"{x},{y},{x2},{y2}" for (x, y, x2, y2) in boxes_draw])
            self.log_writer.writerow([ts, f"{self.fps_smooth:.2f}", len(faces), int(wear), f"{conf:.3f}", bbox_str])

            with self.frame_lock:
                self.last_frame = frame.copy()

            frame_idx += 1

    def _ui_loop(self):
        print("Controls: [q]=退出，[s]=开启/关闭准确率统计(GT模式)，[y]=标注YES，[n]=标注NO")
        if tk is not None:
            root = tk.Tk()
            root.title("Realtime Glasses Controls")

            def on_start():
                if not self.running:
                    idx = int(self.cfg.get("camera_index", 0))
                    self.cap = self._open_camera(idx)
                    if self.cap is None:
                        print("[ERROR] 无法打开摄像头，请尝试更换索引或关闭占用程序。")
                        return
                    self.running = True
                    t = threading.Thread(target=self._loop, daemon=True)
                    t.start()

            def on_stop():
                if self.running:
                    self.running = False
                    time.sleep(0.05)
                    if self.cap:
                        self.cap.release()
                    cv2.destroyAllWindows()

            def toggle_acc():
                self.acc_enabled = not self.acc_enabled
                self.acc_frames = 0
                self.acc_correct = 0

            def run_image_dialog():
                if tk_filedialog is None:
                    print("[WARN] 文件对话框不可用。")
                    return
                path = tk_filedialog.askopenfilename(
                    title="选择图片", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
                )
                if path:
                    threading.Thread(target=self.run_image, args=(Path(path),), daemon=True).start()

            def run_video_dialog():
                if tk_filedialog is None:
                    print("[WARN] 文件对话框不可用。")
                    return
                path = tk_filedialog.askopenfilename(
                    title="选择视频", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.flv")]
                )
                if path:
                    save_path = tk_filedialog.asksaveasfilename(
                        title="可选：保存标注后的视频",
                        defaultextension=".mp4",
                        filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")],
                    )
                    threading.Thread(
                        target=self.run_video, args=(Path(path), Path(save_path) if save_path else None), daemon=True
                    ).start()

            tk.Button(root, text="开始", command=on_start).grid(row=0, column=0, padx=8, pady=8)
            tk.Button(root, text="停止", command=on_stop).grid(row=0, column=1, padx=8, pady=8)
            tk.Button(root, text="切换统计(GT)", command=toggle_acc).grid(row=0, column=2, padx=8, pady=8)
            tk.Button(root, text="图片推理", command=run_image_dialog).grid(row=1, column=0, padx=8, pady=8)
            tk.Button(root, text="视频推理", command=run_video_dialog).grid(row=1, column=1, padx=8, pady=8)

            fps_var = tk.StringVar(value="FPS: 0.0")
            acc_var = tk.StringVar(value="ACC: N/A")
            tk.Label(root, textvariable=fps_var).grid(row=2, column=0, columnspan=3, sticky="w", padx=8)
            tk.Label(root, textvariable=acc_var).grid(row=3, column=0, columnspan=3, sticky="w", padx=8)

            def updater():
                fps_var.set(f"FPS: {self.fps_smooth:.1f}")
                if self.acc_enabled and self.acc_frames > 0:
                    acc = (self.acc_correct / max(1, self.acc_frames)) * 100.0
                    acc_var.set(f"ACC: {acc:.1f}%")
                else:
                    acc_var.set("ACC: N/A")
                root.after(200, updater)

            def render():
                frame = None
                with self.frame_lock:
                    if self.last_frame is not None:
                        frame = self.last_frame.copy()
                if frame is not None:
                    if not self.named:
                        try:
                            cv2.namedWindow("Realtime Glasses", cv2.WINDOW_NORMAL)
                        except Exception:
                            pass
                        self.named = True
                    cv2.imshow("Realtime Glasses", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        time.sleep(0.05)
                        if self.cap:
                            self.cap.release()
                        cv2.destroyAllWindows()
                        root.destroy()
                        return
                    if key == ord('s'):
                        self.acc_enabled = not self.acc_enabled
                        self.acc_frames = 0
                        self.acc_correct = 0
                    if key == ord('y'):
                        self.gt_label = True
                    if key == ord('n'):
                        self.gt_label = False
                root.after(15, render)

            root.after(200, updater)
            root.after(15, render)
            root.mainloop()
        else:
            while self.running:
                frame = None
                with self.frame_lock:
                    if self.last_frame is not None:
                        frame = self.last_frame.copy()
                if frame is not None:
                    if not self.named:
                        try:
                            cv2.namedWindow("Realtime Glasses", cv2.WINDOW_NORMAL)
                        except Exception:
                            pass
                        self.named = True
                    cv2.imshow("Realtime Glasses", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop()
                        break
                    if key == ord('s'):
                        self.acc_enabled = not self.acc_enabled
                        self.acc_frames = 0
                        self.acc_correct = 0
                    if key == ord('y'):
                        self.gt_label = True
                    if key == ord('n'):
                        self.gt_label = False
                time.sleep(0.01)

    def run_image(self, image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] ??????????????????: {image_path}")
            return
        try:
            frame, wear, conf, _ = self._infer_image_no_face(img)
        except Exception as e:
            print(f"[ERROR] ??????????????????: {e}")
            return
        win = "Image Glasses"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print(f"[INFO] ????????????: wear={wear}, conf={conf:.3f}")
        out_dir = Path(self.cfg["output_dir"])
        out_name = f"{image_path.stem}_out{image_path.suffix}"
        out_path = out_dir / out_name
        try:
            cv2.imwrite(str(out_path), frame)
            print(f"[INFO] ????????????: {out_path}")
        except Exception as e:
            print(f"[WARN] ??????????????????: {e}")
        while True:
            cv2.imshow(win, frame)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                break
        cv2.destroyWindow(win)

    def _infer_image_no_face(self, frame: np.ndarray):
        """Direct YOLO on whole image (no face detection)."""
        h, w = frame.shape[:2]
        target_w = self.cfg["frame_width"]
        if w != target_w:
            scale = target_w / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        res = self.model.predict(
            frame,
            imgsz=self.cfg["yolo_imgsz"],
            conf=self.cfg["yolo_conf"],
            iou=self.cfg["yolo_iou"],
            verbose=False,
            device="cpu",
        )
        r0 = res[0]
        wear = False
        conf = 0.0
        boxes_draw = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            b = r0.boxes
            idx = int(np.argmax(b.conf.cpu().numpy()))
            bb = b.xyxy[idx].cpu().numpy().tolist()
            conf = float(b.conf[idx].cpu().numpy())
            wear = True
            boxes_draw.append(tuple(map(int, bb)))

        color = (0, 200, 0) if wear else (0, 0, 255)
        label = "Glasses: YES" if wear else "Glasses: NO"
        for (x0, y0, x1, y1) in boxes_draw:
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Conf: {conf:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2, cv2.LINE_AA)
        return frame, wear, conf, boxes_draw

    def run_video(self, video_path: Path, save_path: Path | None = None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {video_path}")
            return
        writer = None
        win = "Video Glasses"
        if save_path is None:
            out_dir = Path(self.cfg["output_dir"])
            out_name = f"{video_path.stem}_out.mp4"
            save_path = out_dir / out_name
            print(f"[INFO] ????????????????????????: {save_path}")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_out, wear, conf, _ = self._infer_single_frame(frame)
            if save_path is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame_out.shape[:2]
                writer = cv2.VideoWriter(str(save_path), fourcc, cap.get(cv2.CAP_PROP_FPS) or 25.0, (w, h))
            if writer:
                writer.write(frame_out)
            cv2.imshow(win, frame_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        if writer:
            writer.release()
        cv2.destroyWindow(win)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index for realtime mode")
    parser.add_argument("--image", type=Path, help="Run single image inference")
    parser.add_argument("--video", type=Path, help="Run video inference")
    parser.add_argument("--save_video", type=Path, help="Optional path to save annotated video")
    args = parser.parse_args()

    cfg_path = Path("d:/code/glasses/config/realtime.yaml")
    app = RealtimeGlassesApp(cfg_path if cfg_path.exists() else None)
    try:
        if args.image:
            app.run_image(args.image)
        elif args.video:
            app.run_video(args.video, args.save_video)
        else:
            app.start(args.camera)
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
