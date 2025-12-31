# 过滤与训练准备

## 数据过滤
- 最小分辨率：300x300
- 去重方式：phash 近重复阈值=5
- 目标大小：≤500MB，类别均衡（佩戴/不佩戴）
- 分类与检测：YOLO-World `yolov8s-worldv2.pt`，classes=[eyeglasses, glasses, sunglasses]
- 标签格式：YOLO txt，类别0为眼镜（不含置信度列）

## YOLO-World
- 环境：Python venv + CPU Torch + Ultralytics
- 权重选择：`yolov8s-worldv2.pt`（CPU推理更快，精度适中）
- 只检测眼镜：`model.set_classes(["eyeglasses","glasses","sunglasses"])`

## 训练准备
- 切分比例：train 70%，val 20%，test 10%
- 数据结构：`MeGlass_split/{train,val,test}/{images,labels}`
- 数据增强：使用 Ultralytics 默认增强，可在训练时通过 `hsv`, `flipud`, `fliplr`, `scale` 等参数开启
- 评估指标：
  - 检测：Precision/Recall 基于过滤阶段生成的标签
  - 分类：佩戴/不佩戴准确率（有/无检测）
- 基线：在 `test` 子集上使用预训练 YOLO-World 推理得到初始指标

## 运行
- 过滤：`python scripts/glasses_pipeline.py filter --src d:\code\glasses\MeGlass_ori --out d:\code\glasses\MeGlass_filtered --min_w 300 --min_h 300 --max_bytes 500000000 --dedupe_threshold 5`
- 切分：`python scripts/glasses_pipeline.py split --filtered d:\code\glasses\MeGlass_filtered --out d:\code\glasses\MeGlass_split`
- 评估：`python scripts/glasses_pipeline.py evaluate --split d:\code\glasses\MeGlass_split\val`
