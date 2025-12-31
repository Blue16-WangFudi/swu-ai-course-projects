<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

# Xinjiang Dance Action Standard Recognition System
> 中文标题：新疆舞动作规范识别与评估系统
## 概览
- 对新疆舞动作进行规范识别与评估。
- 通常包含视频预处理、关键点提取与时序建模等步骤。
- 输入为视频；输出为动作规范性/标签及汇总结果。
- 团队成员：杨旭伟, 阿卜杜巴日·艾山

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
