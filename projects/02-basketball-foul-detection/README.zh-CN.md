<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

# Basketball Foul Action Detection Using YOLO and State Machine
> 中文标题：基于 YOLO 与状态机的篮球犯规动作智能识别系统
## 概览
- 在比赛视频中识别与犯规相关的动作事件。
- 常见方法将目标/姿态检测（YOLO）与规则/状态机结合。
- 输入为视频；输出为动作标签、时间点或带标注的视频结果。
- 团队成员：李国靖, 蒋文函

## 项目结构
- `code/`：源码
- `data/`：数据集（不随仓库提供；邮件申请）
- `weights/`：权重（不随仓库提供；邮件申请）
- `slides/`：课件（不随仓库提供；邮件申请）
- `tests/`：烟雾测试

## 环境
- 激活环境：

```
conda activate pytorch
```
- 可选依赖（若本项目提供 `requirements.txt`，按需安装）：

```
pip install -r requirements.txt
```
## 获取资源（数据集 / 权重 / 课件）
- 申请邮箱：blue16@email.swu.edu.cn
- 使用政策：
  - 学术研究与教育交流：免费
  - 商业用途（如专利申报、产品化）：需要明确授权
- 本地路径约定：
  - `data/...`
  - `weights/...`

## 快速开始
最小化验证（不需要数据集/权重）：

```
python tests/smoke_test.py
```
## 可复现性说明
- 建议先运行烟雾测试；如项目包含演示/推理脚本，通常位于 `code/`。
- 显存占用与模型规模相关；必要时降低 batch size / 输入分辨率。

## 常见问题
- `Weights not found`：请邮件申请资源并放置到 `weights/`。
- `ModuleNotFoundError`：按需使用 `pip install -r requirements.txt` 安装依赖（若提供）。
- CUDA 相关错误：请确认 PyTorch/CUDA 与显卡驱动匹配。
