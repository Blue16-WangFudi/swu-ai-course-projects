import cv2, pathlib, json, pandas as pd
import mediapipe as mp

# 输入输出路径
video_path = r'D:\py\PythonProject\xinjiangwu\demo\demo.mp4'
out_json = 'keypoints_sequence.json'
out_csv = 'keypoints_sequence.csv'
# 新增：输出视频的路径
out_video = 'annotated_video.mp4'

# 初始化视频读写器
cap = cv2.VideoCapture(video_path)
# 获取原视频的帧率、尺寸等信息，用于创建输出视频
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化VideoWriter，用于写入带有标记的新视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'avc1', 'x264' 等
out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

# 初始化MediaPipe的姿势估计和绘图工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# 可以自定义绘图样式（可选）
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))  # 绿色，BGR格式

all_frames = []

with mp_pose.Pose() as pose:
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 为了绘制，保留一份原始帧的副本
        annotated_frame = frame.copy()

        # 将BGR转换为RGB进行处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb_frame)

        # 提取关键点数据
        lm = [[pt.x, pt.y, pt.z] for pt in res.pose_landmarks.landmark] if res.pose_landmarks else [[None] * 3] * 33
        all_frames.append({'frame': idx, 'landmarks': lm})

        # ------------------------- 关键修改部分 -------------------------
        # 如果检测到关键点，则在 annotated_frame 上绘制它们
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=res.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,  # 绘制关键点之间的连线
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
        # -------------------------------------------------------------

        # 可选：在帧上显示帧编号
        cv2.putText(annotated_frame, f'Frame: {idx}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 将标记好的帧写入输出视频文件
        out.write(annotated_frame)

        # 可选：实时显示处理过程（按'q'退出显示）
        cv2.imshow('Pose Estimation', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        idx += 1

# 释放所有资源
cap.release()
out.release()  # 确保视频写入器被正确关闭
cv2.destroyAllWindows()

print('共提取', len(all_frames), '帧')

# 保存数据文件（与之前相同）
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(all_frames, f, ensure_ascii=False, indent=2)

cols = [f'{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z']]
df = pd.DataFrame([sum(fr['landmarks'], []) for fr in all_frames], columns=cols)
df.insert(0, 'frame', range(len(df)))
df.to_csv(out_csv, index=False, float_format='%.6f')
print('Done!')
print(f'帧数 = {len(df)}')
print(f'带标记的视频已保存至：{out_video}')