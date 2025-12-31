<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

# swu-ai-course-projects

吴鹏程老师开源的课程项目合集（16 个小组项目），已做结构统一与可复现性整理。

- 导师主页：https://ai.swu.edu.cn/info/1071/2528.htm
- 资源申请（数据集 / 权重 / 课件）：blue16@email.swu.edu.cn

## 概览

本仓库收录了面向学术课程的开源项目，涵盖计算机视觉、语音、自然语言处理与智能系统等方向。

为便于快速复现与运行，本仓库统一了：

- `projects/` 下的统一项目结构
- 英文 + 简体中文双语 README（点击即可切换）
- 完整的 `.gitignore` 规则，避免提交大体积二进制文件
- 最小化烟雾测试（优先演示/推理；训练仅文档化，不作为必需验证项）

## 仓库结构

```
projects/
  01-soccer-offside-yolov11/
  ...
  16-autonomous-obstacle-avoidance/
scripts/
  smoke_all.py
```

每个项目遵循：

```
projects/<NN>-<slug>/
  README.md
  README.zh-CN.md
  code/
  data/README.md
  weights/README.md
  slides/README.md
  tests/smoke_test.py
```

## 快速开始

激活已有环境：

```
conda activate pytorch
```

运行全部烟雾测试：

```
python scripts/smoke_all.py
```

按需安装单个项目的可选依赖（仅在需要时安装）：

```
pip install -r projects/<NN>-<slug>/requirements.txt
```

运行单个项目的烟雾测试：

```
python projects/<NN>-<slug>/tests/smoke_test.py
```

## 项目索引

见 `PROJECTS.md`。

## 获取资源（数据集 / 权重 / 课件）

为符合开源仓库体积与分发规范，本仓库不会在 GitHub 中包含这些资源文件。

- 申请邮箱：blue16@email.swu.edu.cn
- 使用政策：
  - 学术研究与教育交流：免费
  - 商业用途（如专利申报、产品化）：需要明确授权

本地路径约定（每个项目）：

- `projects/<NN>-<slug>/data/`
- `projects/<NN>-<slug>/weights/`
- `projects/<NN>-<slug>/slides/`

## 贡献与 Issue

- 贡献指南：`CONTRIBUTING.md`
- 请使用 GitHub 的 Issue 模板提交 Bug / 功能需求。

## 许可证

本仓库使用自定义许可：

- 学术研究与教育交流免费使用
- 商业用途需明确授权
- “按现状”提供，不提供任何担保

详见 `LICENSE`。

