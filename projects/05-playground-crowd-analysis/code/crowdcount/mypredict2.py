from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path

# 1. 加载训练好的最佳模型（替换为你的模型实际路径）
model = YOLO("./playground2_crowd_train/yolo11m_playground2/weights/best.pt")  # 替换为你的best.pt路径

# 2. 计数配置（可根据需求调整）
IOU_THRESH = 0.45  # IOU阈值（NMS去重，避免重复计数）
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_COLOR = (0, 0, 255)  # 红色字体（人数显示）
BOX_COLOR = (0, 255, 0)  # 绿色检测框（框选人体）
BOX_THICKNESS = 2

# 支持的图片格式（可根据需要添加）
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp")


def process_single_image(img_path, output_dir):
    """
    处理单张图片（提取原函数核心逻辑，用于批量调用）
    :param img_path: 单张图片路径
    :param output_dir: 结果图片保存目录
    :return: 计数结果（人数）
    """
    # 读取图片
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"⚠️  图片读取失败（可能损坏）：{img_path}")
        return -1

    # 获取图片尺寸
    img_h, img_w = frame.shape[:2]

    # 模型预测（仅检测person类）
    results = model(
        frame,
        iou=IOU_THRESH,
        classes=[0],  # 仅保留person类（YOLO默认ID=0）
        verbose=False  # 关闭冗余输出
    )

    # 统计人数 + 解析检测框
    crowd_num = 0
    boxes = None
    if results[0].boxes is not None:
        crowd_num = len(results[0].boxes)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # xyxy像素坐标

    # 可视化：绘制检测框和人数
    annotated_frame = frame.copy()
    if crowd_num > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    # 绘制带黑色边框的人数文本（更清晰）
    cv2.putText(annotated_frame, f"Crowd Count: {crowd_num}", (50, 80),
                FONT, FONT_SCALE, (0, 0, 0), thickness=6)
    cv2.putText(annotated_frame, f"Crowd Count: {crowd_num}", (50, 80),
                FONT, FONT_SCALE, FONT_COLOR, thickness=3)

    # 保存结果图片
    img_filename = os.path.basename(img_path)  # 获取原文件名
    save_path = os.path.join(output_dir, img_filename)
    cv2.imwrite(save_path, annotated_frame)

    # 打印当前图片的计数结果
    print(f"✅ {img_filename} - 人群数量：{crowd_num} - 结果保存至：{save_path}")
    return crowd_num


def crowd_count_folder(folder_path, save_output=True, output_dir="crowd_count_results"):
    """
    批量处理文件夹中所有图片的人群计数
    :param folder_path: 目标图片文件夹路径（绝对路径或相对路径）
    :param save_output: 是否保存结果图片（默认True）
    :param output_dir: 结果图片保存目录（默认当前目录下的crowd_count_results）
    """
    # 1. 校验文件夹路径是否存在
    if not os.path.isdir(folder_path):
        print(f"❌ 错误：文件夹路径不存在 → {folder_path}")
        return

    # 2. 创建结果保存目录（如果不存在）
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 结果将保存至：{os.path.abspath(output_dir)}")

    # 3. 遍历文件夹中所有文件，筛选图片格式
    image_paths = []
    for filename in os.listdir(folder_path):
        # 忽略隐藏文件（如.gitignore、.DS_Store等）
        if filename.startswith("."):
            continue
        # 筛选支持的图片格式
        if filename.lower().endswith(SUPPORTED_FORMATS):
            img_path = os.path.join(folder_path, filename)
            image_paths.append(img_path)

    # 4. 检查是否有符合条件的图片
    if len(image_paths) == 0:
        print(f"❌ 错误：文件夹中未找到支持的图片文件（支持格式：{SUPPORTED_FORMATS}）")
        return

    # 5. 批量处理每张图片
    print(f"\n🚀 开始批量处理：共找到 {len(image_paths)} 张图片")
    total_count = 0  # 统计所有图片的总人数（可选）
    success_count = 0  # 成功处理的图片数

    for img_path in image_paths:
        crowd_num = process_single_image(img_path, output_dir if save_output else "")
        if crowd_num >= 0:
            success_count += 1
            total_count += crowd_num  # 累加总人数（不需要可删除）

    # 6. 输出批量处理总结
    print(f"\n📊 批量处理完成！")
    print(f"总计图片数：{len(image_paths)}")
    print(f"成功处理：{success_count} 张")
    print(f"所有图片总人数：{total_count} 人（仅为统计参考）")
    print(f"结果目录：{os.path.abspath(output_dir)}")


# ------------------- 核心调用入口 -------------------
if __name__ == "__main__":
    # ------------------- 配置参数（根据实际修改）-------------------
    TARGET_FOLDER = r"C:\Users\xiong\Desktop\data" # 目标图片文件夹路径
    SAVE_OUTPUT = True  # 是否保存结果图片（True/False）
    OUTPUT_DIR = "playground2_crowd_results"  # 结果保存目录名称

    # ------------------- 调用批量处理函数 -------------------
    crowd_count_folder(
        folder_path=TARGET_FOLDER,
        save_output=SAVE_OUTPUT,
        output_dir=OUTPUT_DIR
    )

    # 可选：处理完后显示任意一张结果图（按ESC关闭）
    result_images = os.listdir(OUTPUT_DIR)
    if result_images:
        sample_img = cv2.imread(os.path.join(OUTPUT_DIR, result_images[0]))
        cv2.imshow("Sample Result", sample_img)
        print("\n提示：按ESC键关闭示例结果窗口")
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()