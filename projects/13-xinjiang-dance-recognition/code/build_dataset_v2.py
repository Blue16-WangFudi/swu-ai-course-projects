"""
build_dataset_v2.py
在单文件脚本基础上只加「遍历+写npy」
python build_dataset_v2.py --dir clips --out_npy npy
"""
import cv2, json, pandas as pd
import mediapipe as mp
import os, pathlib, glob, tqdm, numpy as np
import argparse

mp_pose = mp.solutions.pose

def extract_one_video(video_path):
    """你原来已经跑通的逻辑，只把内部变量包成函数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    all_frames = []
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      smooth_landmarks=True,
                      min_detection_confidence=0.4) as pose:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = [[pt.x, pt.y, pt.z] for pt in res.pose_landmarks.landmark] if res.pose_landmarks else [[0.0]*3]*33
            all_frames.append(lm)
            idx += 1
    cap.release()
    # 转成 (F, 99)  numpy
    return np.array([sum(lst, []) for lst in all_frames], dtype=np.float32)   # (F, 99)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='neg_clips', help='视频目录')
    parser.add_argument('--out_npy', default='neg_npy', help='输出 .npy 目录')
    args = parser.parse_args()

    os.makedirs(args.out_npy, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(args.dir, '*.mp4')) +
                    glob.glob(os.path.join(args.dir, '*.avi')))
    print(f'[INFO] 找到 {len(videos)} 个视频')

    for v in tqdm.tqdm(videos, desc='Extract'):
        seq = extract_one_video(v)
        if seq is None or seq.size == 0:
            print(f'[SKIP] 读取失败: {v}')
            continue
        out_file = os.path.join(args.out_npy,
                                pathlib.Path(v).stem + '.npy')
        np.save(out_file, seq)

    print(f'✔ 全部完成 -> {args.out_npy}/')

if __name__ == '__main__':
    main()