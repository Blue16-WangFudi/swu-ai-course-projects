import warnings
import os
import yaml
import torch
from ultralytics import YOLO

# 忽略警告
warnings.filterwarnings('ignore')


def check_gpu():
    """检查并打印 GPU 信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"\n=================================================")
        print(f"✅ 成功检测到 GPU: {gpu_name}")
        print(f"✅ 显存大小: {mem:.2f} GB")
        print(f"✅ CUDA 版本: {torch.version.cuda}")
        print(f"=================================================\n")
        return '0'
    else:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("❌ 未检测到 GPU！将使用 CPU 训练，速度会非常慢。")
        print("请检查显卡驱动或 PyTorch 版本。")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return 'cpu'


def create_dataset_yaml():
    root_path = r"D:\QQ文件\1"
    if not os.path.exists(root_path):
        print(f"❌ 错误：找不到路径 {root_path}")
        return False

    yaml_content = {
        'path': root_path,
        'train': 'images',
        'val': 'images',
        'names': {0: 'badminton', 1: 'racket'}
    }

    yaml_file = 'badminton.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True)

    return True


def main():
    # 1. 检查 GPU
    device_id = check_gpu()

    # 2. 生成配置
    if not create_dataset_yaml():
        return

    data_cfg = 'badminton.yaml'
    model_cfg = 'yolov8-sea.yaml'
    pretrained_weights = 'yolov8n.pt'

    # 3. 初始化模型
    print(">>> 正在初始化模型...")
    model = YOLO(model_cfg)

    # 加载权重 (如果报错则跳过)
    try:
        model.load(pretrained_weights)
    except Exception:
        print(">>> 注意：预训练权重加载跳过或部分加载，继续训练...")

    # 4. 数据增强参数
    aug_args = {
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 15.0, 'translate': 0.1, 'scale': 0.5,
        'mosaic': 1.0, 'mixup': 0.1, 'fliplr': 0.5,
    }

    # ---------------------------------------------------
    # 第一阶段训练
    # ---------------------------------------------------
    print("\n>>> [阶段1] 开始训练：冻结骨干 (50 Epochs)...")

    # try:
    #     model.train(
    #         data=data_cfg,
    #         epochs=50,
    #         imgsz=640,
    #         batch=8,  # 显存8G建议设为8或16，如果报错改小
    #         device=device_id,  # <--- 强制调用 GPU
    #         project='runs/train',
    #         name='badminton_stage1',
    #         freeze=5,
    #         patience=50,
    #
    #         # --- 关键修复参数 ---
    #         workers=0,  # Windows 必须设为 0，否则易报错
    #         amp=False,  # 关闭 AMP 检查，修复 OSError 报错
    #         # --------------------
    #
    #         **aug_args
    #     )
    # except Exception as e:
    #     print(f"❌ 训练中断: {e}")
    #     return
    #
    # print(">>> 第一阶段完成。")

    # ---------------------------------------------------
    # 第二阶段训练
    # ---------------------------------------------------
    print("\n>>> [阶段2] 开始训练：全网络微调 (100 Epochs)...")

    best_stage1 = r"D:\pythonProject30\runs\train\badminton_stage14\weights\best.pt"

    if os.path.exists(best_stage1):
        model_finetune = YOLO(best_stage1)
        model_finetune.train(
            data=data_cfg,
            epochs=100,
            imgsz=640,
            batch=8,
            device=device_id,  # <--- 强制调用 GPU
            project='runs/train',
            name='badminton_final',
            freeze=0,
            lr0=0.001,

            workers=0,  # Windows 必须设为 0
            amp=False,  # 关闭 AMP

            **aug_args
        )
        print(f">>> 训练全部完成！最佳模型：runs/train/badminton_final/weights/best.pt")
    else:
        print("❌ 未找到第一阶段权重，无法继续。")


if __name__ == '__main__':
    main()