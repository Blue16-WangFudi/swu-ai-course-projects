import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer3DBlock(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 重塑为序列 (B, H*W*D, C)
        B, C, H, W, D = x.shape
        x_seq = x.permute(0, 2, 3, 4, 1).reshape(B, H * W * D, C)

        # Transformer块
        x_seq = x_seq + self.attn(self.norm1(x_seq), self.norm1(x_seq), self.norm1(x_seq))[0]
        x_seq = x_seq + self.mlp(self.norm2(x_seq))

        # 重塑回3D (B, C, H, W, D)
        return x_seq.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)


class OccupancyNetwork3D(nn.Module):

    def __init__(self, in_channels=3, grid_size=(128, 128, 32)):
        super().__init__()
        self.grid_size = grid_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            Transformer3DBlock(64, num_heads=8),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            Transformer3DBlock(128, num_heads=8),
            nn.MaxPool3d(2),
        )

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            Transformer3DBlock(256, num_heads=8),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2),
            Transformer3DBlock(128, num_heads=8),
            nn.ConvTranspose3d(128, 64, 2, stride=2),
            Transformer3DBlock(64, num_heads=8),
            nn.Conv3d(64, 1, 1),  # 输出占用概率
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W, D) 3D特征
        x_enc = self.encoder(x)
        x_bottleneck = self.bottleneck(x_enc)
        occupancy = self.decoder(x_bottleneck)
        return occupancy