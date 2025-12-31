import torch
from kissing_detector import KissingDetector, KissingDetector3DConv
from segmentor import Segmentor


def detect_kissing_from_video(video_path, model_path=None, use_3d=False, min_frames=5, threshold=0.8):
    """
    从视频中检测亲吻片段
    
    参数：
    - video_path: 视频文件路径
    - model_path: 预训练模型权重路径（可选）
    - use_3d: 是否使用3D卷积模型
    - min_frames: 片段的最小帧数
    - threshold: 预测概率阈值
    
    返回：
    - segments: 检测到的亲吻片段列表
    """
    # 初始化模型
    if use_3d:
        model = KissingDetector3DConv(num_classes=2, feature_extract=False, use_vggish=True)
    else:
        model = KissingDetector(conv_model_name='resnet', num_classes=2, feature_extract=False, use_vggish=True)
    
    # 加载预训练模型权重（如果提供）
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    # 设置模型为评估模式
    model.eval()
    
    # 初始化Segmentor
    segmentor = Segmentor(model=model, min_frames=min_frames, threshold=threshold)
    
    # 检测亲吻片段
    segments = segmentor.get_segments(video_path)
    
    return segments


def visualize_segments_from_video(video_path, model_path=None, use_3d=False, min_frames=5, threshold=0.8, n_to_show=10):
    """
    可视化视频中的亲吻片段
    
    参数：
    - video_path: 视频文件路径
    - model_path: 预训练模型权重路径（可选）
    - use_3d: 是否使用3D卷积模型
    - min_frames: 片段的最小帧数
    - threshold: 预测概率阈值
    - n_to_show: 每个片段显示的图像数量
    """
    # 初始化模型
    if use_3d:
        model = KissingDetector3DConv(num_classes=2, feature_extract=False, use_vggish=True)
    else:
        model = KissingDetector(conv_model_name='resnet', num_classes=2, feature_extract=False, use_vggish=True)
    
    # 加载预训练模型权重（如果提供）
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    # 设置模型为评估模式
    model.eval()
    
    # 初始化Segmentor
    segmentor = Segmentor(model=model, min_frames=min_frames, threshold=threshold)
    
    # 可视化亲吻片段
    segmentor.visualize_segments(video_path, n_to_show=n_to_show)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法：")
        print("python detect_kissing.py <视频文件路径> [模型路径] [是否使用3D模型(0/1)]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    use_3d = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    
    print("开始检测视频中的亲吻片段...")
    print(f"视频路径: {video_path}")
    print(f"模型路径: {model_path}")
    print(f"是否使用3D模型: {use_3d}")
    
    # 可视化结果
    visualize_segments_from_video(video_path, model_path, use_3d)
