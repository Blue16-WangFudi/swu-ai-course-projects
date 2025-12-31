# 改编自 http://cs231n.github.io/assignments2019/assignment3/
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage.filters import gaussian_filter1d
from torch import nn

import params
from pipeline import BuildDataset


class QualitativeAnalysis:
    def __init__(self,
                 model: nn.Module,
                 img_size: int,
                 videos_and_frames: Dict[str, List[int]],
                 class_names: Dict[int, str]):
        self.class_names = class_names
        self.img_size = img_size
        self.model = model
        self.videos_and_frames = videos_and_frames

        self.features = {
            vid: BuildDataset.one_video_extract_audio_and_stills(vid)
            for vid in self.videos_and_frames}

    # 方法改编自斯坦福大学cs231n作业3，可在以下网址获取：
    # http://cs231n.github.io/assignments2019/assignment3/
    @staticmethod
    def _compute_saliency_maps(A, I, y, model):
        """
        使用模型为图像X和标签y计算类显著性图。

        输入：
        - A: 输入音频；形状为 (N, 1, 96, 64) 的张量
        - I: 输入图像；形状为 (N, 3, H, W) 的张量
        - y: X的标签；形状为 (N,) 的LongTensor
        - model: 用于计算显著性图的预训练CNN。

        返回：
        - saliency: 形状为 (N, H, W) 的张量，给出输入图像的显著性图。
        """
        # 确保模型处于"test"模式
        model.eval()

        # 使输入张量需要梯度
        # A.requires_grad_()
        I.requires_grad_()

        scores = model(A, I).gather(1, y.view(-1, 1)).squeeze()
        scores.backward(torch.ones(scores.size()))
        saliency, _ = torch.max(I.grad.abs(), dim=1)

        return saliency

    # 同样改编自cs231n作业3
    def _show_saliency_maps(self, A, I, y):
        # 将X和y从numpy数组转换为Torch张量
        I_tensor = torch.cat([
            BuildDataset.transformer(self.img_size)(Image.fromarray(i)).unsqueeze(0)
            for i in I], dim=0)
        A_tensor = torch.cat([a.unsqueeze(0) for a in A])
        y_tensor = torch.LongTensor(y)

        # 为X中的图像计算显著性图
        saliency = self._compute_saliency_maps(A_tensor, I_tensor, y_tensor, self.model)

        # 将显著性图从Torch张量转换为numpy数组，并一起显示图像和显著性图。
        saliency = saliency.numpy()
        N = len(I)
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.imshow(I[i])
            plt.axis('off')
            plt.title(self.class_names[y[i]])
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.hot)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.show()

    @staticmethod
    def _img_transform_reverse_to_np(x: torch.Tensor) -> np.array:
        rev = BuildDataset.transform_reverse(x)
        return np.array(rev)

    def saliency_maps(self):
        for vid, indices in self.videos_and_frames.items():
            A = [self.features[vid][0][idx] for idx in indices]
            I = [self._img_transform_reverse_to_np(self.features[vid][1][idx])
                 for idx in indices]
            y = [1 if 'kissing' in vid else 0] * len(A)
            self._show_saliency_maps(A, I, y)
            print('=' * 10)

    # 接下来的几个方法取自cs231n
    @staticmethod
    def jitter(X, ox, oy):
        """
        随机抖动图像的辅助函数。

        输入
        - X: 形状为 (N, C, H, W) 的 PyTorch 张量
        - ox, oy: 沿 W 和 H 轴抖动的像素数整数

        返回：形状为 (N, C, H, W) 的新 PyTorch 张量
        """
        if ox != 0:
            left = X[:, :, :, :-ox]
            right = X[:, :, :, -ox:]
            X = torch.cat([right, left], dim=3)
        if oy != 0:
            top = X[:, :, :-oy]
            bottom = X[:, :, -oy:]
            X = torch.cat([bottom, top], dim=2)
        return X

    @staticmethod
    def _blur_image(X, sigma=1):
        X_np = X.cpu().clone().numpy()
        X_np = gaussian_filter1d(X_np, sigma, axis=2)
        X_np = gaussian_filter1d(X_np, sigma, axis=3)
        X.copy_(torch.Tensor(X_np).type_as(X))
        return X

    def create_class_visualization(self, target_y, model, dtype, a, **kwargs):
        """
        生成一个图像，以最大化预训练模型下target_y的分数。

        输入：
        - target_y: 范围在 [0, 1000) 内的整数，给出类的索引
        - model: 用于生成图像的预训练CNN
        - dtype: 用于计算的Torch数据类型

        关键字参数：
        - l2_reg: 图像上L2正则化的强度
        - learning_rate: 采取多大的步长
        - num_iterations: 使用多少次迭代
        - blur_every: 多久模糊一次图像作为隐式正则化器
        - max_jitter: 将图像抖动多少作为隐式正则化器
        - show_every: 多久显示一次中间结果
        """

        def deprocess(x):
            return BuildDataset.transform_reverse(x.squeeze(0))

        model.type(dtype)
        l2_reg = kwargs.pop('l2_reg', 1e-3)
        learning_rate = kwargs.pop('learning_rate', 25)
        num_iterations = kwargs.pop('num_iterations', 100)
        blur_every = kwargs.pop('blur_every', 10)
        max_jitter = kwargs.pop('max_jitter', 16)
        show_every = kwargs.pop('show_every', 25)

        # 将图像随机初始化为PyTorch张量，并使其需要梯度。
        img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype).requires_grad_()

        for t in range(num_iterations):
            # 随机抖动图像一点；这会产生稍微好一点的结果
            ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
            img.data.copy_(self.jitter(img.data, ox, oy))

            target = model(a, img)[0, target_y]
            target.backward()
            g = img.grad.data
            g -= 2 * l2_reg * img.data
            img.data += learning_rate * (g / g.norm())
            img.grad.zero_()

            # 撤销随机抖动
            img.data.copy_(self.jitter(img.data, -ox, -oy))

            # 作为正则化器，限制并定期模糊图像
            for c in range(3):
                lo = float(-params.mean[c] / params.std[c])
                hi = float((1.0 - params.mean[c]) / params.std[c])
                img.data[:, c].clamp_(min=lo, max=hi)
            if t % blur_every == 0:
                self._blur_image(img.data, sigma=0.5)

            # 定期显示图像
            if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
                plt.imshow(deprocess(img.data.clone().cpu()))
                class_name = self.class_names[target_y]
                plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
                plt.gcf().set_size_inches(4, 4)
                plt.axis('off')
                plt.show()

        return deprocess(img.data.cpu())
