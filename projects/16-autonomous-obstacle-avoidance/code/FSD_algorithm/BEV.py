import torch
from torch import nn


class LiftSplatShoot(nn.Module):


    def __init__(self, feature_dim=256, bev_dim=200, bev_res=0.5):
        super().__init__()
        self.bev_dim = bev_dim
        self.bev_res = bev_res

        # 深度分布预测头
        self.depth_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 42, 1),  # D个深度bin
            nn.Softmax(dim=1)
        )

        # BEV编码器
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

    def lift(self, features, intrinsics):
        """将图像特征提升到3D空间"""
        B, C, H, W = features.shape

        # 预测深度分布
        depth_prob = self.depth_head(features)  # (B, D, H, W)

        # 生成3D点云
        # 简化实现，实际需要相机内外参
        return features, depth_prob

    def splat(self, features_3d):
        """将3D特征投影到BEV平面"""
        # 将特征累加到BEV网格
        bev_features = torch.zeros(
            features_3d.shape[0],
            features_3d.shape[1],
            self.bev_dim,
            self.bev_dim
        ).to(features_3d.device)

        # 简化：均匀投影
        # 实际需要精确的几何投影
        return bev_features

    def forward(self, features, intrinsics):
        features_3d, depth_prob = self.lift(features, intrinsics)
        bev_features = self.splat(features_3d)
        bev_features = self.bev_encoder(bev_features)
        return bev_features, depth_prob