from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np

USE_WEBCAM = True  # required by the original implementation
CAMERA_ID = 0  # 0 = default webcam
SAVE_OUTPUT = False  # whether to save recording

# Fatigue detection settings
INIT_DURATION_SEC = 2.0
EYE_RATIO = 0.4
LIGHT_EYE_SEC = 0.6
SEVERE_EYE_SEC = 1.5

MOUTH_OPEN_THRESHOLD = 7
YAWN_DURATION_SEC = 2.0

# Head pose configuration
face_3d_model = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip (1)
        (0.0, -330.0, -65.0),  # chin (152)
        (-225.0, 170.0, -135.0),  # left eye outer corner (263)
        (225.0, 170.0, -135.0),  # right eye outer corner (33)
        (-150.0, -150.0, -125.0),  # left mouth corner (291)
        (150.0, -150.0, -125.0),  # right mouth corner (61)
    ],
    dtype=np.float64,
)

face_2d_landmarks_ids = [1, 152, 263, 33, 291, 61]
HEAD_DOWN_THRESHOLD = 25
HEAD_NOD_DURATION_SEC = 2.0


def _create_face_mesh() -> mp.solutions.face_mesh.FaceMesh:
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def main() -> int:
    """Webcam demo entrypoint.

    NOTE: This file is intentionally import-safe so the repository smoke tests
    can import the module without opening a webcam or entering an infinite loop.
    """

    face_mesh = _create_face_mesh()
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Unable to open webcam. Please check permissions or device.")
        face_mesh.close()
        return 1

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = "fatigue_detection_recording.mp4"
    out = None
    if SAVE_OUTPUT:
        out = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    # Convert time thresholds to frame counts.
    init_frames = int(fps * INIT_DURATION_SEC)
    light_eye_frames = max(1, int(fps * LIGHT_EYE_SEC))
    severe_eye_frames = max(1, int(fps * SEVERE_EYE_SEC))
    yawn_frames = max(1, int(fps * YAWN_DURATION_SEC))
    head_down_frames = max(1, int(fps * HEAD_NOD_DURATION_SEC))

    # State variables
    frame_count = 0
    open_eye_heights: list[float] = []
    ref_eye_height: float | None = None
    consecutive_eye_closed = 0.0
    consecutive_mouth_open = 0.0
    consecutive_head_down = 0.0
    is_initialized = False

    print("Starting real-time fatigue detection...")
    print(f"Calibrating for {INIT_DURATION_SEC:.1f}s, then running live detection.")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Webcam read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            eye_height = 0.0
            mouth_height = 0.0
            is_eye_closed = False
            is_mouth_open_long = False
            is_head_down = False
            pitch = yaw = roll = 0.0

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                # Eye height
                left_eye_h = abs(lm[386].y - lm[374].y) * h
                right_eye_h = abs(lm[159].y - lm[145].y) * h
                eye_height = (left_eye_h + right_eye_h) / 2

                # Mouth height
                mouth_height = abs(lm[14].y - lm[13].y) * h

                # Calibration stage
                if not is_initialized and frame_count < init_frames:
                    open_eye_heights.append(eye_height)
                    cv2.putText(
                        frame,
                        f"[CALIBRATING... {frame_count}/{init_frames}]",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2,
                    )

                # Detection stage
                else:
                    if not is_initialized and open_eye_heights:
                        ref_eye_height = max(open_eye_heights)
                        is_initialized = True
                        print(f"Calibration done. ref open-eye height = {ref_eye_height:.1f}px")

                    if is_initialized and ref_eye_height is not None:
                        # Eye closure
                        eye_threshold = ref_eye_height * EYE_RATIO
                        is_eye_closed = eye_height < eye_threshold

                        # Yawn detection
                        if mouth_height > MOUTH_OPEN_THRESHOLD:
                            consecutive_mouth_open += 1
                        else:
                            consecutive_mouth_open = max(0, consecutive_mouth_open - 0.7)
                        is_mouth_open_long = consecutive_mouth_open >= yawn_frames

                        # Head pose estimation
                        face_2d = []
                        for idx in face_2d_landmarks_ids:
                            x = lm[idx].x * w
                            y = lm[idx].y * h
                            face_2d.append([x, y])
                        face_2d = np.array(face_2d, dtype=np.float64)

                        focal_length = w
                        center = (w / 2, h / 2)
                        camera_matrix = np.array(
                            [
                                [focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1],
                            ],
                            dtype=np.float64,
                        )
                        dist_coeffs = np.zeros((4, 1))

                        success, rotation_vec, translation_vec = cv2.solvePnP(
                            face_3d_model, face_2d, camera_matrix, dist_coeffs
                        )

                        if success:
                            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                            pose_mat = np.hstack((rotation_mat, translation_vec))
                            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                                pose_mat
                            )
                            pitch, yaw, roll = [angle[0] for angle in euler_angles]
                            pitch = (pitch + 180) % 360 - 180
                            yaw = (yaw + 180) % 360 - 180
                            roll = (roll + 180) % 360 - 180
                            is_head_down = pitch > HEAD_DOWN_THRESHOLD

                        # Draw key landmarks
                        for idx in [386, 374, 159, 145]:
                            x = int(lm[idx].x * w)
                            y = int(lm[idx].y * h)
                            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                        for idx in [13, 14]:
                            x = int(lm[idx].x * w)
                            y = int(lm[idx].y * h)
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Update counters (after calibration)
            if is_initialized:
                if is_eye_closed:
                    consecutive_eye_closed += 1
                else:
                    consecutive_eye_closed = max(0, consecutive_eye_closed - 0.7)

                if is_head_down:
                    consecutive_head_down += 1
                else:
                    consecutive_head_down = max(0, consecutive_head_down - 0.7)

            # Fatigue decision
            severe_eye = consecutive_eye_closed >= severe_eye_frames
            light_eye = consecutive_eye_closed >= light_eye_frames
            yawn_detected = is_mouth_open_long
            head_nod_fatigue = consecutive_head_down >= head_down_frames

            if yawn_detected or severe_eye or head_nod_fatigue:
                status = "SEVERE FATIGUE"
                color = (0, 0, 255)
            elif light_eye:
                status = "LIGHT FATIGUE (Eyes)"
                color = (0, 255, 255)
            else:
                status = "NORMAL"
                color = (0, 255, 0)

            # Overlay info
            if is_initialized and ref_eye_height is not None:
                cv2.putText(
                    frame,
                    f"Eye Ref: {ref_eye_height:.1f}px",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Closed Eye Frms: {consecutive_eye_closed:.1f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Mouth Ht: {mouth_height:.1f}px",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Open Mouth Frms: {consecutive_mouth_open:.1f}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Pitch: {pitch:.1f}°",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Head Down Frms: {consecutive_head_down:.1f}",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                status,
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
            )

            if SAVE_OUTPUT and out:
                out.write(frame)

            cv2.imshow("Fatigue Detection - LIVE (Eye+Mouth+Head)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

    finally:
        cap.release()
        if SAVE_OUTPUT and out:
            out.release()
            print(f"Saved recording: {os.path.abspath(output_video)}")
        cv2.destroyAllWindows()
        face_mesh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
