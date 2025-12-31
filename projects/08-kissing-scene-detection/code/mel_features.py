# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""定义从音频波形计算梅尔频谱特征的例程。"""

import numpy as np


def frame(data, window_length, hop_length):
    """将数组转换为连续可能重叠的帧序列。
    形状为(num_samples, ...)的n维数组被转换为形状为(num_frames, window_length, ...)的(n+1)维数组，
    其中每个帧在前一个帧之后hop_length个点开始。
    这是使用stride_tricks实现的，因此不会复制原始数据。但是，没有零填充，
    所以末尾的任何不完整帧都不会被包含。
    参数:
      data: 维度N >= 1的np.array。
      window_length: 每个帧中的样本数。
      hop_length: 每个窗口之间的步长（以样本为单位）。
    返回:
      (N+1)-D np.array，行数等于可以提取的完整帧数。
    """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
    """计算"周期性"汉宁窗。
    经典的汉宁窗定义为一个从零开始和结束的升余弦，除了奇数长度窗口的中间点外，每个值出现两次。
    Matlab称此为"对称"窗口，np.hanning()返回此窗口。然而，对于傅里叶分析，
    这实际上表示周期为N-1的余弦的略多于一个周期，因此不能在长度为N的傅里叶基上紧凑地表示。
    相反，最好使用一个刚好在最终零值之前结束的升余弦 - 即周期为N的余弦的完整周期。
    Matlab称此为"周期性"窗口。此例程计算它。
    参数:
      window_length: 返回窗口中的点数。
    返回:
      包含周期性汉宁窗的1D np.array。
    """
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                               np.arange(window_length)))


def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
    """计算短时傅里叶变换幅度。
    参数:
      signal: 输入时域信号的1D np.array。
      fft_length: 要应用的FFT大小。
      hop_length: 传递给FFT的每个帧之间的步长（以样本为单位）。
      window_length: 传递给FFT的每个样本块的长度。
    返回:
      2D np.array，其中每行包含对应输入样本帧的FFT的fft_length/2+1个唯一值的幅度。
    """
    frames = frame(signal, window_length, hop_length)
    # 对每个帧应用帧窗口。我们使用周期性汉宁窗（周期为window_length的余弦）
    # 而不是np.hanning的对称汉宁窗（周期为window_length-1）。
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


# 梅尔频谱常量和函数。
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
    """使用HTK公式将频率转换为梅尔刻度。
    参数:
      frequencies_hertz: 标量或np.array，单位为赫兹。
    返回:
      与frequencies_hertz大小相同的对象，包含梅尔刻度上的对应值。
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
    """返回一个可以后乘频谱行以生成梅尔频谱的矩阵。
    返回一个np.array矩阵A，可用于后乘形状为frames x bins的频谱值（STFT幅度）矩阵S，
    生成形状为frames x num_mel_bins的"梅尔频谱"M。即M = S A。
    经典的HTK算法利用相邻梅尔带的互补性，将每个FFT bin乘以一个梅尔权重，
    然后用正负号将其添加到该bin贡献的两个相邻梅尔带中。在这里，通过将此操作表示为矩阵乘法，
    我们从每帧num_fft次乘法（加上约2*num_fft次加法）变为约num_fft^2次乘法和加法。
    然而，由于这些都可能在一次np.dot()调用中完成，因此不清楚哪种方法在Python中更快。
    矩阵乘法的吸引力在于更通用、更灵活，并且更容易阅读。
    参数:
      num_mel_bins: 生成的梅尔频谱中的带数。这是输出矩阵的列数。
      num_spectrogram_bins: 源频谱数据中的bin数，理解为fft_size/2 + 1，
        即频谱仅包含非冗余的FFT bin。
      audio_sample_rate: 输入到频谱的音频的每秒采样数。我们需要这个来计算每个频谱bin的实际频率，
        这决定了它们如何映射到梅尔刻度。
      lower_edge_hertz: 要包含在梅尔频谱中的频率的下限。这对应于最低三角带的下边缘。
      upper_edge_hertz: 最高频率带的期望上边缘。
    返回:
      形状为(num_spectrogram_bins, num_mel_bins)的np.array。
    引发:
      ValueError: 如果频率边缘顺序错误或超出范围。
    """
    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f必须 >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                         (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f大于奈奎斯特频率 %.1f" %
                         (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    # 第i个梅尔带（从i=1开始）的中心频率为band_edges_mel[i]，下边缘为band_edges_mel[i-1]，
    # 上边缘为band_edges_mel[i+1]。因此，我们需要band_edges_mel数组中有num_mel_bins + 2个值。
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                                 hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    # 用于后乘特征数组的矩阵，其行是num_spectrogram_bins个频谱值。
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        # 计算每个频谱bin的下斜率和上斜率。
        # 线段在*梅尔*域中是线性的，而不是在赫兹域中。
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                       (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                       (upper_edge_mel - center_mel))
        # ..然后将它们与彼此和零相交。
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                              upper_slope))
    # HTK排除频谱DC bin；确保它始终获得零系数。
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def log_mel_spectrogram(data,
                        audio_sample_rate=8000,
                        log_offset=0.0,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        **kwargs):
    """将波形转换为对数幅度梅尔频率频谱图。
    参数:
      data: 波形数据的1D np.array。
      audio_sample_rate: 数据的采样率。
      log_offset: 取对数时添加到此值以避免-Infs。
      window_length_secs: 每个分析窗口的持续时间。
      hop_length_secs: 连续分析窗口之间的步长。
      **kwargs: 传递给spectrogram_to_mel_matrix的其他参数。
    返回:
      形状为(num_frames, num_mel_bins)的2D np.array，包含连续帧的对数梅尔滤波器组幅度。
    """
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    spectrogram = stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples)
    mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(
        num_spectrogram_bins=spectrogram.shape[1],
        audio_sample_rate=audio_sample_rate, **kwargs))
    return np.log(mel_spectrogram + log_offset)
