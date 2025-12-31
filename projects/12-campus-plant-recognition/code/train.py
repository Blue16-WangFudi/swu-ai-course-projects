import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# =============================
# 1. 数据路径
# =============================
data_dir = "dataset11"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# 检查数据路径
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"训练数据路径不存在: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"验证数据路径不存在: {val_dir}")

# =============================
# 2. 数据增强
# =============================
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# =============================
# 3. 类别信息
# =============================
num_classes = len(train_dataset.classes)
print("=" * 50)
print(f"类别数量：{num_classes}")
print(f"类别名称：{train_dataset.classes}")
print(f"训练样本数：{len(train_dataset)}")
print(f"验证样本数：{len(val_dataset)}")
print("=" * 50)

# =============================
# 4. TensorBoard 日志
# =============================
log_dir = r"C:\Users\xiaolian\tensorboard_logs\plant_classification"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard 日志路径: {log_dir}")
print("提示：运行 tensorboard --logdir={log_dir} 查看训练日志")

# =============================
# 5. ResNet50 + 冻结前面层
# =============================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 冻结除layer4和fc外的所有层
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# 替换全连接层
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
if torch.cuda.is_available():
    print(f"GPU名称：{torch.cuda.get_device_name(0)}")
model = model.to(device)

# =============================
# 6. 损失函数 & 优化器
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# =============================
# 7. 训练记录初始化
# =============================
epochs = 20
train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = 0.0

# =============================
# 8. 训练循环
# =============================
print("\n开始训练...")
print("=" * 50)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 30)

    # -------- 训练阶段 --------
    model.train()
    train_loss = 0.0
    train_correct = 0

    train_bar = tqdm(train_loader, desc=f"Training", ncols=100)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 计算训练集指标
    train_loss_avg = train_loss / len(train_dataset)
    train_acc = train_correct.double() / len(train_dataset)
    train_losses.append(train_loss_avg)
    train_accs.append(train_acc.item())

    # -------- 验证阶段 --------
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Validation", ncols=100)
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

            val_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 计算验证集指标
    val_loss_avg = val_loss / len(val_dataset)
    val_acc = val_correct.double() / len(val_dataset)
    val_losses.append(val_loss_avg)
    val_accs.append(val_acc.item())

    # 学习率调度
    scheduler.step()

    # 打印结果
    print(f"\n训练 Loss: {train_loss_avg:.4f} | 训练 Acc: {train_acc:.4f}")
    print(f"验证 Loss: {val_loss_avg:.4f} | 验证 Acc: {val_acc:.4f}")
    print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }, "best_plant_classifier_resnet50.pth")
        print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")

    # TensorBoard记录
    writer.add_scalar("Loss/train", train_loss_avg, epoch)
    writer.add_scalar("Loss/val", val_loss_avg, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

# =============================
# 9. 训练结束 - 绘制最终曲线
# =============================
print("\n" + "=" * 50)
print("训练完成！开始绘制结果曲线...")

# 创建图表 - 同时显示损失和准确率
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
epochs_range = range(1, epochs + 1)

# 绘制损失曲线
ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
ax1.set_title('Training and Validation Loss', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_xticks(epochs_range)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_xlim(1, epochs)

# 绘制准确率曲线
ax2.plot(epochs_range, train_accs, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
ax2.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
ax2.set_title('Training and Validation Accuracy', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_xticks(epochs_range)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')
ax2.set_xlim(1, epochs)

# 调整子图间距
plt.tight_layout()

# 保存高分辨率图片
plt.savefig("training_results.png", dpi=300, bbox_inches='tight')
print("结果曲线已保存为: training_results.png")

# 显示图片（可选）
plt.show()

# =============================
# 10. 保存最终模型和训练记录
# =============================
# 保存最终模型
torch.save(model.state_dict(), "plant_classifier_resnet50_final.pth")

# 保存训练记录
np.savez('training_history.npz',
         train_losses=train_losses,
         val_losses=val_losses,
         train_accs=train_accs,
         val_accs=val_accs)

# 打印总结信息
print("\n训练总结:")
print(f"最佳验证准确率: {best_val_acc:.4f}")
print(f"最终训练损失: {train_losses[-1]:.4f}")
print(f"最终验证损失: {val_losses[-1]:.4f}")
print(f"最终训练准确率: {train_accs[-1]:.4f}")
print(f"最终验证准确率: {val_accs[-1]:.4f}")
print("\n文件保存:")
print("1. 最佳模型: best_plant_classifier_resnet50.pth")
print("2. 最终模型: plant_classifier_resnet50_final.pth")
print("3. 训练结果曲线: training_results.png")
print("4. 训练记录: training_history.npz")

# 关闭TensorBoard writer
writer.close()