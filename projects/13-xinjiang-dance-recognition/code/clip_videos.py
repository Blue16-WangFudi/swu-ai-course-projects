"""
clip_videos.py
剪切长视频并统一输出帧率为 30 fps
python clip_videos.py --dir data --out_dir clips --mode sliding --win 3 --step 1
"""
import cv2, os, glob, argparse, subprocess
from tqdm import tqdm
import numpy as np

# =========== 帧率统一工具 ===========
def resample_video_ffmpeg(in_path, out_path, fps=30):
    """用 ffmpeg 离线转帧率，画质最好"""
    cmd = [
        'ffmpeg', '-y', '-i', in_path,
        '-r', str(fps),           # 输出帧率
        '-c:a', 'copy',           # 音频直接拷贝
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def resample_video_cv2(in_path, out_path, fps=30):
    """无 ffmpeg 时的退路：OpenCV 逐帧重采样"""
    cap = cv2.VideoCapture(in_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / src_fps
    tgt_frames = int(dur * fps)
    idx = np.linspace(0, total - 1, tgt_frames, dtype=int)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fid in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frm = cap.read()
        if not ret:
            break
        writer.write(frm)
    writer.release()
    cap.release()
    return out_path

def unify_fps(in_path, tmp_dir, fps=30):
    """先整段转 30 fps，返回新路径"""
    os.makedirs(tmp_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(in_path))[0] + '_30fps.mp4'
    out_path = os.path.join(tmp_dir, basename)
    # 优先 ffmpeg，失败再退到 cv2
    try:
        resample_video_ffmpeg(in_path, out_path, fps)
    except FileNotFoundError:
        print('[WARN] ffmpeg 未找到，改用 OpenCV 重采样...')
        resample_video_cv2(in_path, out_path, fps)
    return out_path
# =====================================

def sliding_window(in_path, out_dir, win_sec, step_sec, fps_out=30):
    cap = cv2.VideoCapture(in_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win_frames = int(win_sec * fps_out)
    step_frames = int(step_sec * fps_out)
    basename = os.path.splitext(os.path.basename(in_path))[0].replace('_30fps', '')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    idx = 0
    for start in range(0, total_frames - win_frames + 1, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        out_path = os.path.join(out_dir, f'{basename}_clip{idx:03d}.mp4')
        writer = cv2.VideoWriter(out_path, fourcc, fps_out,
                                 (int(cap.get(3)), int(cap.get(4))))
        for _ in range(win_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        idx += 1
    cap.release()

def crop_fixed(in_path, out_dir, start_sec, duration_sec, fps_out=30):
    cap = cv2.VideoCapture(in_path)
    start_frame = int(start_sec * fps_out)
    end_frame = start_frame + int(duration_sec * fps_out)
    basename = os.path.splitext(os.path.basename(in_path))[0].replace('_30fps', '')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out_dir, f'{basename}_fixed.mp4')
    writer = cv2.VideoWriter(out_path, fourcc, fps_out,
                             (int(cap.get(3)), int(cap.get(4))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()
    cap.release()

def crop_with_pose(in_path, out_dir, win_sec, fps_out=30):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        min_detection_confidence=0.4)
    cap = cv2.VideoCapture(in_path)
    win_frames = int(win_sec * fps_out)
    has_person = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        has_person.append(int(res.pose_landmarks is not None))
    pose.close()
    cap.release()

    has_person = np.array(has_person)
    changes = np.diff(np.pad(has_person, (1, 1), 'constant'))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) == 0:
        print(f'[WARN] {in_path} 未检测到人体，跳过')
        return
    best_i = np.argmax(ends - starts)
    start_frame, end_frame = starts[best_i], ends[best_i]
    n_clips = (end_frame - start_frame) // win_frames
    if n_clips == 0:
        print(f'[WARN] {in_path} 有效段 < {win_sec}s，跳过')
        return

    basename = os.path.splitext(os.path.basename(in_path))[0].replace('_30fps', '')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(in_path)
    for c in range(n_clips):
        s = start_frame + c * win_frames
        e = s + win_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        out_path = os.path.join(out_dir, f'{basename}_pose{c:03d}.mp4')
        writer = cv2.VideoWriter(out_path, fourcc, fps_out,
                                 (int(cap.get(3)), int(cap.get(4))))
        for _ in range(win_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='data', help='原始长视频目录')
    parser.add_argument('--out_dir', default='clips', help='输出片段目录')
    parser.add_argument('--mode', choices=['fixed', 'sliding', 'pose'], default='sliding')
    parser.add_argument('--win', type=float, default=3, help='片段时长(s)')
    parser.add_argument('--step', type=float, default=1, help='滑动窗口步长(s)')
    parser.add_argument('--start', type=float, default=0, help='固定裁剪起点(s)')
    parser.add_argument('--duration', type=float, default=3, help='固定裁剪时长(s)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    videos = glob.glob(os.path.join(args.dir, '*.mp4')) + \
             glob.glob(os.path.join(args.dir, '*.avi'))
    tmp_dir = os.path.join(args.out_dir, '_tmp_30fps')
    for v in tqdm(videos, desc='1. 统一帧率→2. 剪切'):
        # 先整段转 30 fps
        resampled = unify_fps(v, tmp_dir, fps=30)
        # 再按指定模式剪切
        if args.mode == 'fixed':
            crop_fixed(resampled, args.out_dir, args.start, args.duration)
        elif args.mode == 'sliding':
            sliding_window(resampled, args.out_dir, args.win, args.step)
        else:
            crop_with_pose(resampled, args.out_dir, args.win)
    # 可选：删除临时文件
    # import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
    print('✔ 全部裁剪完成 ->', args.out_dir)

if __name__ == '__main__':
    main()