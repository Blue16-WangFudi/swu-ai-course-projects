import matplotlib.pyplot as plt

# 你的训练日志数据（粘贴进去即可）
losses = [
    119.5697, 107.9539, 105.1479, 97.7773, 95.0429, 96.4054, 93.4499, 94.4237,
    92.1324, 88.3403, 88.3663, 88.8120, 89.5758, 86.1199, 88.6027, 87.2851,
    85.6426, 89.0199, 85.5740, 89.2331
]

test_acc = [
    67.17, 67.42, 75.94, 76.19, 75.69, 75.69, 77.19, 76.19, 76.94, 77.69,
    76.44, 70.68, 76.94, 77.69, 73.18, 78.95, 78.70, 75.94, 76.94, 77.94
]

epochs = list(range(1, len(losses) + 1))

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve.png", dpi=300)
plt.show()

# Plot Test Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, test_acc, marker='s')
plt.title("Test Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(60, 85)
plt.grid(True)
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

print("✓ 曲线已绘制并保存为 loss_curve.png 和 accuracy_curve.png")
