# 改编自 https://github.com/harritaylor/torchvggish
from typing import Tuple

import torch.nn as nn
from torch import hub

import conv

VGGISH_WEIGHTS = (
    # "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish-cbfe8f1c.pth"
    'https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish-918c2d05.pth'
)
PCA_PARAMS = (
    "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish_pca_params-4d878af3.npz"
)


class VGGishParams:
    """
    这些参数不应该被修改。它们被添加到这个文件中是为了方便使用。
    """

    NUM_FRAMES = (96,)  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.

    # 用于特征和样本生成的超参数。
    SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

    # 用于嵌入后处理的参数。
    PCA_EIGEN_VECTORS_NAME = "pca_eigen_vectors"
    PCA_MEANS_NAME = "pca_means"
    QUANTIZE_MIN_VAL = -2.0
    QUANTIZE_MAX_VAL = +2.0


"""
VGGish
输入: 96x64 1通道 spectrogram
输出: 128维嵌入
"""


class VGGish(nn.Module):
    def __init__(self, feature_extract: bool):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, VGGishParams.NUM_BANDS, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(VGGishParams.NUM_BANDS, VGGishParams.EMBEDDING_SIZE, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, VGGishParams.EMBEDDING_SIZE),
            nn.ReLU(inplace=True),
        )
        conv.set_parameter_requires_grad(self.features, feature_extract)
        conv.set_parameter_requires_grad(self.embeddings, feature_extract)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        return x


def vggish(feature_extract: bool) -> Tuple[VGGish, int]:
    """
    VGGish是Tensorflow的VGGish架构的PyTorch实现，用于为Audioset创建嵌入。
    它生成96ms音频片段的128维嵌入。始终使用预训练模型。
    """
    model = VGGish(feature_extract)
    model.load_state_dict(hub.load_state_dict_from_url(VGGISH_WEIGHTS), strict=True)
    return model, VGGishParams.EMBEDDING_SIZE
