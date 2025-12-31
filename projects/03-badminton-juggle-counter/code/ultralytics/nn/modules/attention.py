# In ultralytics/nn/modules/attention.py

# ... (文件已有的其他代码) ...

# 在文件末尾添加以下代码
import torch
import torch.nn as nn

class SEAttention(nn.Module):
    def __init__(self, c1, r=16):  # c1: input channels, r: reduction ratio
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // r, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)