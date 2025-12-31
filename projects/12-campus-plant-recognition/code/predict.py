import torch
from torchvision import models, datasets, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================
# 1. 基础配置（解决中文显示+设备/路径）
# =============================
# 解决Matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['PingFang SC']    # macOS
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 当前目录路径
current_dir = os.getcwd()
print(f"当前识别目录：{current_dir}")

# =============================
# 2. 加载类别列表（与训练一致）
# =============================
try:
    train_dataset = datasets.ImageFolder("dataset/train")
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {classes}")
except Exception as e:
    print(f"读取类别失败（请确认dataset/train目录存在）：{e}")
    # 若训练集路径不可用，手动指定类别（替换为你的实际类别）
    classes = ["香椿叶", "银杏叶", "枫叶", "梧桐叶"]  # 示例类别，需修改
    num_classes = len(classes)

# =============================
# 3. 数据预处理（与训练一致）
# =============================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准化
                         std=[0.229, 0.224, 0.225])
])


# =============================
# 4. 加载ResNet50模型
# =============================
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # 加载训练好的模型权重
    try:
        model.load_state_dict(torch.load("plant_classifier_resnet50.pth", map_location=device))
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败：{e}")
        exit()
    model = model.to(device)
    model.eval()  # 推理模式
    return model


model = load_model()


# =============================
# 5. 核心函数：预测+可视化展示
# =============================
def predict_and_show(image_path):
    """
    预测单张图片，并可视化展示（标注预测结果和置信度）
    :param image_path: 图片路径
    """
    try:
        # 读取图片
        image = Image.open(image_path).convert("RGB")
        # 预处理
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 模型预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        # 解析结果
        pred_class = classes[pred_idx.item()]
        confidence = conf.item() * 100

        # =============================
        # 可视化展示（核心）
        # =============================
        plt.figure(figsize=(10, 8))  # 设置画布大小
        plt.imshow(image)
        plt.axis('off')  # 关闭坐标轴

        # 添加预测结果文本（带背景框，提升可读性）
        text_content = f"预测类别：{pred_class}\n置信度：{confidence:.2f}%"
        # 文本样式：白色字体+黑色背景+半透明
        plt.text(
            10, 30,  # 文本位置（左上角）
            text_content,
            fontsize=16,
            color='white',
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor='black',
                alpha=0.7  # 背景透明度
            )
        )

        # 设置标题（图片名称）
        img_name = os.path.basename(image_path)
        plt.title(f"识别图片：{img_name}", fontsize=18, pad=10)

        # 调整布局，避免文本被裁剪
        plt.tight_layout()
        # 显示图片
        plt.show()

        # 打印控制台信息
        print(f"\n✅ 图片：{img_name}")
        print(f"🔍 预测类别：{pred_class}")
        print(f"📊 置信度：{confidence:.2f}%")
        print("-" * 60)

        return pred_class, confidence

    except Exception as e:
        print(f"\n❌ 处理图片 {image_path} 失败：{e}")
        print("-" * 60)
        return None, None


# =============================
# 6. 批量识别当前目录所有图片
# =============================
def batch_process_current_dir():
    """遍历当前目录所有图片，逐个预测并展示"""
    # 支持的图片格式
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    # 统计识别数量
    processed_count = 0

    print("\n" + "=" * 60)
    print("开始识别当前目录下的图片...")
    print("=" * 60)

    # 遍历当前目录文件
    for file_name in os.listdir(current_dir):
        # 过滤非图片文件
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext not in img_extensions:
            continue

        # 拼接完整路径
        file_path = os.path.join(current_dir, file_name)
        # 跳过目录（防止误判）
        if os.path.isdir(file_path):
            continue

        # 预测并展示
        predict_and_show(file_path)
        processed_count += 1

    # 识别完成统计
    print("\n" + "=" * 60)
    print(f"识别完成！共处理 {processed_count} 张图片")
    print("=" * 60)


# =============================
# 7. 执行批量识别
# =============================
if __name__ == "__main__":
    batch_process_current_dir()