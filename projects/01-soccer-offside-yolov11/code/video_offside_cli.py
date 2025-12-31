import argparse
import cv2
from pathlib import Path

from GetField import get_field
from PersonAndPose import get_person_boxes, get_person_keypoints
from Play_Team_Cluster import cluster_players_by_jersey
from Vanishing_Points import (
    get_horizontal_vanishing_point,
    get_vertical_vanishing_point,
    draw_vanishing_points
)
from OffSide_Judge import Caculate_angle, isOffSide


def deduplicate_events(frames, stride):
    """去掉相邻帧的重复事件，只保留起始帧。"""
    if not frames:
        return []
    uniq = []
    prev = -stride * 2
    for f in sorted(frames):
        if f - prev > stride * 2:
            uniq.append(f)
            prev = f
    return uniq


def process_video(
    video_path: str,
    off_team: int,
    def_team: int,
    output_dir: str = "video_results",
    frame_stride: int = 5,  # 默认每5帧处理一次，提升性能
    cache_interval: int = 30,  # 增加消失点缓存时间，减少计算
    save_offside_frames: bool = True,
    on_progress=None,
    on_frame=None,
    preview_stride: int = 10,
    on_offside_extended=None,  # 新增：越位连线图回调
):
    """
    逐帧检测视频越位。
    Args:
        video_path: 视频路径
        off_team: 进攻方队伍编号
        def_team: 防守方队伍编号
        output_dir: 结果输出目录
        frame_stride: 每隔多少帧处理一次
        cache_interval: 消失点缓存间隔（帧）
        save_offside_frames: 是否保存检测到越位的帧
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    og_out_dir = Path(output_dir)/Path(video_path).stem
    og_out_dir.mkdir(parents=True, exist_ok=True)

    vanishing_cache = {"frame": -cache_interval * 2, "value": None}
    offside_frames = []
    # 用于跨帧一致性跟踪的参考聚类中心
    reference_cluster_centers = None
    # 添加一个标志来记录是否已经发送过第一次越位连线图
    first_offside_sent = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_stride > 1 and frame_id % frame_stride != 0:
            continue
        out_dir = og_out_dir/Path(f'{frame_id}')
        out_dir.mkdir(parents=True, exist_ok=True)
        # 场地检测（使用缓存减少重复计算）
        game_area = get_field(frame, str(out_dir), show=False, save_mask=False)
        if game_area is None or isinstance(game_area, int):
            continue

        # 人物检测和关键点提取（不保存中间结果以提升性能）
        person_boxes = get_person_boxes(game_area, frame)
        # 视频处理时不保存中间结果以提升性能
        person_keypoints, image = get_person_keypoints(game_area, frame, str(out_dir), person_boxes=person_boxes, save_result=False)
        if not person_keypoints:
            continue

        # 队伍分类（不保存可视化结果以提升性能）
        # 视频处理时不保存队伍分类可视化结果以提升性能
        # 使用参考聚类中心保持跨帧一致性
        result = cluster_players_by_jersey(person_keypoints, image, str(out_dir), save_result=False, reference_centers=reference_cluster_centers)
        if len(result) == 3:
            labeled_players, _, cluster_centers = result
            # 更新参考聚类中心（使用第一帧或定期更新）
            if reference_cluster_centers is None or frame_id % (cache_interval * 2) == 0:
                reference_cluster_centers = cluster_centers
        else:
            labeled_players = result[0]
        if not labeled_players:
            continue

        # 消失点缓存，避免逐帧重复计算
        if (
            vanishing_cache["value"] is None
            or frame_id - vanishing_cache["frame"] > cache_interval
        ):
            try:
                hor_vp = get_horizontal_vanishing_point(game_area)
                side = "right" if hor_vp[0] > 0 else "left"
                ver_vp = get_vertical_vanishing_point(game_area, side)
                vanishing_cache = {"frame": frame_id, "value": (side, ver_vp)}
            except Exception:
                # 保持上一次的缓存；若无缓存则跳过
                if vanishing_cache["value"] is None:
                    continue

        side, ver_vp = vanishing_cache["value"]

        # 计算最靠近底线的点与角度
        Caculate_angle(labeled_players, ver_vp, side)
        # 画出连线图
        extended_img = draw_vanishing_points(frame,ver_vp,labeled_players,str(out_dir))
        is_offside = bool(isOffSide(labeled_players, off_team, def_team))

        if is_offside:
            offside_frames.append(frame_id)
            # 新增：第一次检测到越位时，发送连线图
            if not first_offside_sent and on_offside_extended:
                try:
                    # 创建连线图预览（可以缩放以减小尺寸）
                    preview_extended = extended_img.copy()
                    # 只进行必要的压缩，如果图像太大
                    if preview_extended.shape[1] > 1200:  # 宽度过大才压缩
                        scale = 1200 / preview_extended.shape[1]
                        new_width = 1200
                        new_height = int(preview_extended.shape[0] * scale)
                        preview_extended = cv2.resize(preview_extended, (new_width, new_height),
                                                      interpolation=cv2.INTER_LANCZOS4)
                    on_offside_extended(frame_id, preview_extended)
                    first_offside_sent = True
                except Exception as e:
                    print(f"发送连线图失败: {e}")

            if save_offside_frames:
                annotated = frame.copy()
                for player in labeled_players:
                    if player.get("team_label") == off_team and player.get("most_point"):
                        x, y = map(int, player["most_point"])
                        cv2.circle(annotated, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f"OFFSIDE frame {frame_id}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                cv2.imwrite(str(out_dir / f"offside_{frame_id:06d}.jpg"), annotated)

        # 进度回调
        if on_progress:
            try:
                on_progress(frame_id, total_frames)
            except Exception:
                pass

        # 预览回调：首帧/采样帧/越位帧
        if on_frame and (frame_id <= 1 or frame_id % preview_stride == 0 or is_offside):
            try:
                annotated_preview = frame.copy()
                # 队伍颜色映射
                team_colors = {
                    -2: (128, 128, 128),  # 未分类
                    -1: (0, 255, 0),      # 噪声/裁判
                    0: (255, 0, 0),       # Team 0 - 蓝色 (BGR格式)
                    1: (0, 0, 255),       # Team 1 - 红色 (BGR格式)
                }
                
                for player in labeled_players:
                    bbox = player.get("bbox")
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        team_label = player.get("team_label", -2)
                        color = team_colors.get(team_label, (0, 255, 255))
                        cv2.rectangle(annotated_preview, (x1, y1), (x2, y2), color, 2)
                        
                        # 显示队伍编号标签
                        if team_label >= 0:
                            tag = f"Team {team_label}"
                        elif team_label == -1:
                            tag = "Ref"
                        else:
                            tag = "Unclassified"
                        
                        # 绘制标签背景
                        label_size = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        cv2.rectangle(annotated_preview, (x1, y1 - 22), (x1 + label_size[0] + 8, y1), color, -1)
                        cv2.putText(annotated_preview, tag, (x1 + 4, y1 - 6), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # 绘制最靠近底线的点
                    if player.get("most_point"):
                        x, y = map(int, player["most_point"])
                        cv2.circle(annotated_preview, (x, y), 6, (0, 255, 0), -1)
                
                # 显示帧信息和越位状态
                status_text = f"Frame {frame_id} | Offside: {'Yes' if is_offside else 'No'}"
                cv2.putText(
                    annotated_preview,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if not is_offside else (0, 0, 255),
                    2,
                )
                
                # 显示队伍编号说明（如果已设置）
                if off_team is not None and def_team is not None:
                    legend_text = f"Off: Team {off_team} (Blue) | Def: Team {def_team} (Red)"
                    cv2.putText(
                        annotated_preview,
                        legend_text,
                        (20, annotated_preview.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                
                on_frame(frame_id, annotated_preview, is_offside)
            except Exception as e:
                print(f"预览帧绘制失败: {e}")
                pass

    cap.release()

    unique_events = deduplicate_events(offside_frames, frame_stride)
    return {
        "total_frames": total_frames,
        "fps": fps,
        "offside_frames": offside_frames,
        "unique_events": unique_events,
        "output_dir": str(og_out_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="视频越位检测（命令行版）")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--off", required=True, type=int, help="进攻方队伍编号")
    parser.add_argument("--defn", required=True, type=int, help="防守方队伍编号")
    parser.add_argument("--stride", type=int, default=3, help="帧采样步长，默认每3帧处理1帧")
    parser.add_argument("--out", default="video_results", help="输出目录")
    parser.add_argument("--no-save", action="store_true", help="不保存越位帧截图")
    args = parser.parse_args()

    stats = process_video(
        video_path=args.video,
        off_team=args.off,
        def_team=args.defn,
        output_dir=args.out,
        frame_stride=max(1, args.stride),
        save_offside_frames=not args.no_save,
    )

    print(f"视频总帧数: {stats['total_frames']} | FPS: {stats['fps']:.2f}")
    if stats["unique_events"]:
        print(f"检测到 {len(stats['unique_events'])} 次越位事件，起始帧: {stats['unique_events']}")
        print(f"越位帧截图目录: {stats['output_dir']}")
    else:
        print("未检测到越位事件")


if __name__ == "__main__":
    main()

