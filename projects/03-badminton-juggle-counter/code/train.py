import warnings
import os
import yaml
import torch
from ultralytics import YOLO

# 忽略警告
warnings.filterwarnings('ignore')


def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n✅ 成功检测到 GPU: {gpu_name}")
        return 0
    else:
        print("\n❌ 未检测到 GPU！")
        return 'cpu'


def create_nano_p2_yaml():


    yaml_file = 'yolov8n-p2.yaml'
    
    return yaml_file


def create_dataset_yaml():
    root_path = "badminton_split_dataset"
    if not os.path.exists(root_path):
        print(f"❌ 路径错误: {root_path}")
        return False

    yaml_content = {
        'path': root_path,
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'badminton', 1: 'racket'}
    }
    with open('badminton_split.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    return 'badminton_split.yaml'


def main():
    device = check_gpu()
    data_cfg = create_dataset_yaml()
    model_cfg = create_nano_p2_yaml()

    print(f"\n>>> 1. 初始化 YOLOv8n-P2 模型...")
    # 这一步构建网络结构
    model = YOLO(model_cfg)

    print(f">>> 2. 加载预训练权重 (迁移学习)...")
    # ❗❗❗ 关键修正：加载 v8n.pt，忽略不匹配的层
    # 第一次运行会自动下载 yolov8n.pt
    try:
        model.load('yolov8n.pt')
        print("✅ 预训练权重加载成功！Backbone部分已迁移。")
    except Exception as e:
        print(f"⚠️ 权重加载警告: {e}")
        print("将尝试直接开始训练...")

    # 针对 4060 (8G) 的极限调优参数
        # 针对 RTX 4060 (8G) 的修正版参数
    train_args = {
            'imgsz': 1280,  # 📉 降级：1280 -> 960 (这是8G显存不开AMP的极限)
            'batch': 4,  # 📉 降级：极限压缩 Batch，防止瞬间峰值溢出
            'epochs': 150,  # 保持不变
            'patience': 50,

            # ⚠️ 换回 SGD，因为它比 AdamW 省显存
            'optimizer': 'SGD',
            'lr0': 0.01,  # SGD 的初始学习率

            'hsv_h': 0.015,
            'hsv_s': 0.5,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.3,

            'amp': False,  # 保持关闭，避免报错
            'workers': 0,  # 保持0
            'cache': False,  # 关闭缓存
    }

    print("\n>>> 开始训练 (Nano-P2 + 1280px)...")
    try:
        model.train(
            data=data_cfg,
            device=device,
            project='runs/train',
            name='badminton_nano_p2_1280',
            **train_args
        )
        print(f">>> 训练完成！")
    except Exception as e:
        print(f"\n❌ 训练错误: {e}")
        if "CUDA out of memory" in str(e):
            print("\n⚠️ 显存不足建议：")
            print("1. 将 batch 改为 4")
            print("2. 将 imgsz 改为 1024")


if __name__ == '__main__':
    main()