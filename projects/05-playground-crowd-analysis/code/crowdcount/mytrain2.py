from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")
    results = model.train(
        # 基础配置
        data=r"D:\deeplearning\ultralytics-8.3.163\datasets\playground_crowd.yaml",
        epochs=40,
        batch=8,
        lr0=0.001,
        imgsz=640,
        device=0,

        # 数据增强配置
        augment=True,  # 启用数据增强
        hsv_h=0.015,  # 色调增强强度
        hsv_s=0.7,  # 饱和度增强强度
        hsv_v=0.4,  # 亮度增强强度
        translate=0.1,  # 平移增强
        scale=0.5,  # 缩放增强
        fliplr=0.5,  # 水平翻转概率
        mosaic=0.5,  # Mosaic增强概率（YOLO11建议降低）

        # 模型和训练配置
        pretrained=True,  # 使用预训练权重
        single_cls=True,  # 单类别训练
        patience=5,  # 早停耐心值
        save=True,  # 保存模型
        project="playground2_crowd_train",  # 项目目录
        name="yolo11m_playground2",  # 实验名称

        # 优化器配置
        optimizer="SGD",  # 优化器
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 热身轮数
        warmup_momentum=0.8,  # 热身动量
        warmup_bias_lr=0.1,  # 热身偏置学习率

        # 数据加载配置
        workers=8,  # 数据加载线程数
        rect=False,  # 矩形训练
        cache=False,  # 缓存数据集

        # 验证配置
        val=True,  # 训练中验证
        plots=True,  # 保存训练图表

        # 模型保存配置
        exist_ok=True,  # 覆盖已有目录
        resume=False,  # 恢复训练
    )

# 最佳模型将保存在：./playground2_crowd_train/yolo11s_playground2/weights/best.pt