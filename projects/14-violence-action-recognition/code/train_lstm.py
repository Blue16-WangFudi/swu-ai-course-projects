import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from lstm_model import FightLSTM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载数据
X, y = torch.load("data/dataset.pt")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("y unique:", torch.unique(y))

dataset = TensorDataset(X, y)

# 划分训练集和测试集（80%训练, 20%测试）
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 模型、优化器、损失
model = FightLSTM().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 20
best_acc = 0.0

# 新增：记录训练过程
loss_history = []
acc_history = []

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)

        optim.zero_grad()
        output = model(seqs)
        loss = criterion(output, labels)
        loss.backward()
        optim.step()

        total_loss += loss.item()

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            output = model(seqs)
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total

    loss_history.append(total_loss)
    acc_history.append(acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss={total_loss:.4f}, Test Acc={acc*100:.2f}%")

    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "lstm_model_best.pth")

print("✔ 模型训练完成，最佳模型已保存为 lstm_model_best.pth")

# 绘制损失曲线
plt.figure()
plt.plot(loss_history, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training Loss Curve")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

# 绘制精度曲线
plt.figure()
plt.plot([a * 100 for a in acc_history], label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("LSTM Test Accuracy Curve")
plt.legend()
plt.savefig("acc_curve.png")
plt.close()

# 输出混淆矩阵和指标
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for seqs, labels in test_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        output = model(seqs)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\n====== Confusion Matrix ======")
print(cm)

print("\n====== Classification Report ======")
print(classification_report(all_labels, all_preds, target_names=["Non-violence", "Violence"]))

# 绘制混淆矩阵图
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-violence", "Violence"],
            yticklabels=["Non-violence", "Violence"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("LSTM Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("✔ 已保存: loss_curve.png, acc_curve.png, confusion_matrix.png")
