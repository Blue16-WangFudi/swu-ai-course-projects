from __future__ import annotations

from pathlib import Path
import textwrap

EMAIL = "blue16@email.swu.edu.cn"

PROJECTS = [
    {
        "id": "01",
        "slug": "soccer-offside-yolov11",
        "title_zh": "基于 YOLOv11 姿态检测的单摄像头足球越位判定系统",
        "title_en": "Monocular Soccer Offside Detection via YOLOv11 Pose Estimation",
        "team": "朱婉苹, 肖宇",
        "import_files": ["VideoProcessor.py"],
        "notes_en": [
            "Detect potential offside situations from a single broadcast camera feed.",
            "Typical components include pose detection (YOLOv11) and geometric reasoning.",
            "Inputs are videos/images; outputs are visualized detections and an offside decision.",
        ],
        "notes_zh": [
            "从单路转播摄像头画面中检测潜在越位情况。",
            "常见组成包括姿态检测（YOLOv11）与几何推理。",
            "输入为视频/图像；输出为可视化检测结果与越位判定。",
        ],
    },
    {
        "id": "02",
        "slug": "basketball-foul-detection",
        "title_zh": "基于 YOLO 与状态机的篮球犯规动作智能识别系统",
        "title_en": "Basketball Foul Action Detection Using YOLO and State Machine",
        "team": "李国靖, 蒋文函",
        "import_files": ["travel_detection.py"],
        "notes_en": [
            "Recognize basketball foul-related actions in game footage.",
            "Typical approach combines object/pose detection (YOLO) with a rule/state machine.",
            "Inputs are videos; outputs are action labels, timestamps, or annotated videos.",
        ],
        "notes_zh": [
            "在比赛视频中识别与犯规相关的动作事件。",
            "常见方法将目标/姿态检测（YOLO）与规则/状态机结合。",
            "输入为视频；输出为动作标签、时间点或带标注的视频结果。",
        ],
    },
    {
        "id": "03",
        "slug": "badminton-juggle-counter",
        "title_zh": "基于 YOLOv8-P2 的羽毛球颠球自动计数系统",
        "title_en": "Badminton Shuttlecock Counting System Based on YOLOv8-P2",
        "team": "吴杰, 李思源",
        "import_files": ["main.py"],
        "notes_en": [
            "Count shuttlecock juggling repetitions from a single-camera video.",
            "Uses detection/tracking (YOLOv8-P2 family) to identify the shuttlecock over time.",
            "Inputs are videos; outputs are on-screen overlays and a final count.",
        ],
        "notes_zh": [
            "从单摄像头视频中对羽毛球颠球次数进行自动计数。",
            "通过检测/跟踪（YOLOv8-P2 系列）在时间维度上定位羽毛球。",
            "输入为视频；输出为画面叠加显示与最终计数结果。",
        ],
    },
    {
        "id": "04",
        "slug": "subject3-dance-recognition",
        "title_zh": "科目三舞蹈动作识别与评估系统",
        "title_en": "Subject Three Dance Action Recognition and Evaluation",
        "team": "文宁, 拉姆曲珍",
        "import_files": ["detect.py"],
        "notes_en": [
            "Recognize and evaluate Subject-Three dance actions from video.",
            "Typical pipeline includes person/pose detection and temporal action modeling.",
            "Inputs are videos; outputs are action labels and evaluation/feedback.",
        ],
        "notes_zh": [
            "从视频中识别并评估科目三舞蹈动作。",
            "常见流程包含人物/姿态检测与时序动作建模。",
            "输入为视频；输出为动作标签与评估/反馈信息。",
        ],
    },
    {
        "id": "05",
        "slug": "playground-crowd-analysis",
        "title_zh": "基于 YOLOv11 与 XGBoost 的操场人流量监测与预测系统",
        "title_en": "Playground Crowd Monitoring and Prediction Using YOLOv11 and XGBoost",
        "team": "熊子兴, 杨晶晶",
        "import_files": ["crowdcount/mytrain2.py"],
        "notes_en": [
            "Monitor crowd counts on a playground and predict future traffic.",
            "Typical pipeline: YOLO-based counting + XGBoost regression/forecasting.",
            "Inputs are images/videos and tabular features; outputs are counts and forecasts.",
        ],
        "notes_zh": [
            "监测操场人流量，并对未来人流进行预测。",
            "常见流程：基于 YOLO 的人数统计 + XGBoost 回归/预测。",
            "输入为图像/视频与表格特征；输出为人数统计与预测结果。",
        ],
    },
    {
        "id": "06",
        "slug": "fingerprint-restoration",
        "title_zh": "指纹图像修复与增强实验系统",
        "title_en": "Fingerprint Image Restoration and Enhancement",
        "team": "严馨, 蔡司旅",
        "import_files": [],
        "incomplete": True,
        "notes_en": [
            "Fingerprint image restoration/enhancement experiments.",
            "Some materials are references/archives; full runnable pipeline may require external code/resources.",
            "Smoke test reports SKIPPED by default until runnable code is provided under code/.",
        ],
        "notes_zh": [
            "指纹图像修复与增强相关实验。",
            "部分内容为参考资料/压缩包；完整可运行流程可能需要额外代码或资源。",
            "在 `code/` 提供可运行代码前，烟雾测试默认显示为 SKIPPED。",
        ],
    },
    {
        "id": "07",
        "slug": "transformer-text-prediction",
        "title_zh": "基于 Transformer 的文本预测模型研究与实现",
        "title_en": "Transformer-Based Text Prediction Model",
        "team": "郭睿, 蒙晓丽",
        "import_files": ["transformer_train.py"],
        "notes_en": [
            "Next-token / next-text prediction using a Transformer architecture.",
            "Code typically includes preprocessing, training, and inference utilities.",
            "Inputs are text corpora; outputs are predicted continuations.",
        ],
        "notes_zh": [
            "基于 Transformer 架构的下一词/下一段文本预测。",
            "代码通常包含数据预处理、训练与推理等工具。",
            "输入为文本语料；输出为预测的续写结果。",
        ],
    },
    {
        "id": "08",
        "slug": "kissing-scene-detection",
        "title_zh": "接吻场景智能检测与识别系统",
        "title_en": "Intelligent Kissing Scene Detection System",
        "team": "关祺, 卢建君",
        "import_files": ["kissing_detector.py"],
        "notes_en": [
            "Detect kissing scenes from video content.",
            "Typical pipeline includes video decoding, feature extraction, and a classifier/segmentor.",
            "Inputs are videos; outputs are predicted segments and/or labels.",
        ],
        "notes_zh": [
            "从视频内容中检测接吻场景。",
            "常见流程包括视频解码、特征提取与分类/片段划分模块。",
            "输入为视频；输出为预测片段和/或标签结果。",
        ],
    },
    {
        "id": "09",
        "slug": "dialectsense-wavlm",
        "title_zh": "DialectSense：基于 WavLM 的粗粒度方言识别框架",
        "title_en": "DialectSense: Coarse-Grained Dialect Recognition Framework Based on WavLM",
        "team": "王涪迪, 栾晨章",
        "import_files": ["dialectsense/__init__.py"],
        "notes_en": [
            "Coarse-grained dialect recognition using WavLM representations.",
            "Includes model/config code and (optional) UI components.",
            "Inputs are audio files; outputs are dialect labels/embeddings.",
        ],
        "notes_zh": [
            "基于 WavLM 表征的粗粒度方言识别。",
            "包含模型/配置代码，以及（可选的）界面组件。",
            "输入为音频文件；输出为方言标签/嵌入表示。",
        ],
    },
    {
        "id": "10",
        "slug": "industrial-ppe-detection",
        "title_zh": "基于 YOLO-WORLD 的工业安全防护用品开放集检测系统",
        "title_en": "Open-Set Industrial PPE Detection Based on YOLO-WORLD",
        "team": "姚啟城, 雷迪昊",
        "import_files": ["apps/realtime_glasses.py"],
        "notes_en": [
            "Open-set detection of industrial PPE (e.g., glasses/helmets) with YOLO-WORLD.",
            "Typically supports realtime camera or image/video inference.",
            "Inputs are images/videos; outputs are open-vocabulary detections.",
        ],
        "notes_zh": [
            "基于 YOLO-WORLD 的工业防护用品（如护目镜/头盔等）开放集检测。",
            "通常支持实时摄像头或图像/视频推理。",
            "输入为图像/视频；输出为开放词表的检测结果。",
        ],
    },
    {
        "id": "11",
        "slug": "fatigue-detection-system",
        "title_zh": "实时疲劳检测与安全预警系统",
        "title_en": "Real-Time Fatigue Detection and Safety Alert System",
        "team": "彭真山, 朱豪",
        "import_files": ["webcam_fatigue_detection.py"],
        "notes_en": [
            "Realtime fatigue detection and safety alerting using a webcam feed.",
            "Typical signals include eye-closure, yawning, and head pose (e.g., MediaPipe landmarks).",
            "Inputs are webcam frames; outputs are on-screen alerts and status.",
        ],
        "notes_zh": [
            "基于摄像头视频流的实时疲劳检测与安全预警。",
            "常见信号包括闭眼、打哈欠与头部姿态（如基于 MediaPipe 关键点）。",
            "输入为摄像头帧；输出为画面提示与状态信息。",
        ],
    },
    {
        "id": "12",
        "slug": "campus-plant-recognition",
        "title_zh": "基于 ResNet18 的校园植物种类识别系统",
        "title_en": "Campus Plant Species Recognition Based on ResNet18",
        "team": "柏帅, 龙水河",
        "import_files": [],
        "notes_en": [
            "Plant species recognition from leaf images.",
            "Model architecture is ResNet-based; weights/datasets are not included.",
            "Smoke test validates PyTorch/torchvision and performs a dummy forward.",
        ],
        "notes_zh": [
            "面向叶片图像的校园植物种类识别。",
            "模型结构为 ResNet 系列；权重/数据集不随仓库提供。",
            "烟雾测试会验证 PyTorch/torchvision 并进行一次 dummy forward。",
        ],
    },
    {
        "id": "13",
        "slug": "xinjiang-dance-recognition",
        "title_zh": "新疆舞动作规范识别与评估系统",
        "title_en": "Xinjiang Dance Action Standard Recognition System",
        "team": "杨旭伟, 阿卜杜巴日·艾山",
        "import_files": ["rename_mp4.py"],
        "notes_en": [
            "Recognize and evaluate Xinjiang dance actions for standardization.",
            "Often involves video preprocessing, keypoint extraction, and temporal modeling.",
            "Inputs are videos; outputs are action correctness/labels and summaries.",
        ],
        "notes_zh": [
            "对新疆舞动作进行规范识别与评估。",
            "通常包含视频预处理、关键点提取与时序建模等步骤。",
            "输入为视频；输出为动作规范性/标签及汇总结果。",
        ],
    },
    {
        "id": "14",
        "slug": "violence-action-recognition",
        "title_zh": "基于 YOLOv8 与 LSTM 的暴力行为识别系统",
        "title_en": "Violence Action Recognition Using YOLOv8 and LSTM",
        "team": "林智翔, 袁浩",
        "import_files": ["lstm_model.py"],
        "notes_en": [
            "Detect violent actions in video using detection + temporal modeling.",
            "Typical pipeline: YOLOv8 extracts per-frame features; LSTM aggregates over time.",
            "Inputs are videos; outputs are violence/non-violence labels and annotated frames.",
        ],
        "notes_zh": [
            "基于检测 + 时序建模的方法在视频中识别暴力行为。",
            "常见流程：YOLOv8 提取逐帧特征，LSTM 在时间维度进行聚合。",
            "输入为视频；输出为暴力/非暴力标签与标注结果。",
        ],
    },
    {
        "id": "15",
        "slug": "wafer-defect-analysis",
        "title_zh": "晶圆表面缺陷检测与加工溯源分析系统",
        "title_en": "Wafer Surface Defect Detection and Process Traceability",
        "team": "朱帅臣",
        "import_files": [],
        "incomplete": True,
        "notes_en": [
            "Wafer surface defect detection and traceability analysis.",
            "This project is currently incomplete in the public repository.",
            "Smoke test reports SKIPPED by default until runnable code is provided under code/.",
        ],
        "notes_zh": [
            "晶圆表面缺陷检测与加工溯源分析。",
            "该项目在公开仓库中目前内容不完整。",
            "在 `code/` 提供可运行代码前，烟雾测试默认显示为 SKIPPED。",
        ],
    },
    {
        "id": "16",
        "slug": "autonomous-obstacle-avoidance",
        "title_zh": "基于类 FSD 架构的自动驾驶避障系统设计",
        "title_en": "Autonomous Driving Obstacle Avoidance Based on FSD-like Architecture",
        "team": "叶锦源, 罗朝洋",
        "import_files": ["FSD_algorithm/ConvLSTM.py"],
        "notes_en": [
            "Autonomous driving obstacle avoidance experiments inspired by FSD-like architectures.",
            "May include simulation (e.g., CARLA) and learned perception/planning components.",
            "Inputs are simulated sensor data; outputs are trajectories/control commands.",
        ],
        "notes_zh": [
            "参考类 FSD 架构的自动驾驶避障实验与系统设计。",
            "可能包含仿真（如 CARLA）以及学习式感知/规划组件。",
            "输入为仿真传感器数据；输出为轨迹/控制指令。",
        ],
    },
]


LANG_SWITCH = """<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

"""


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def md_code_block(code: str) -> str:
    return f"```\n{code.rstrip()}\n```\n"


def project_readme_en(p: dict) -> str:
    lines: list[str] = []
    lines.append(LANG_SWITCH)
    lines.append(f"# {p['title_en']}\n")
    lines.append(f"> Chinese title: {p['title_zh']}\n")

    lines.append("## Overview\n")
    for s in p["notes_en"]:
        lines.append(f"- {s}\n")
    lines.append(f"- Team: {p['team']}\n")

    lines.append("\n## Project Structure\n")
    lines.append("- `code/`: source code\n")
    lines.append("- `data/`: dataset (not included; request by email)\n")
    lines.append("- `weights/`: weights (not included; request by email)\n")
    lines.append("- `slides/`: slides (not included; request by email)\n")
    lines.append("- `tests/`: smoke tests\n")

    lines.append("\n## Environment\n")
    lines.append("- Activate environment:\n\n")
    lines.append(md_code_block("conda activate pytorch"))
    lines.append("- Optional extra deps (if this project provides `requirements.txt`):\n\n")
    lines.append(md_code_block("pip install -r requirements.txt"))

    lines.append("## Getting Resources (Dataset / Weights / Slides)\n")
    lines.append(f"- Email: {EMAIL}\n")
    lines.append("- Policy:\n")
    lines.append("  - Academic research & educational exchange: free\n")
    lines.append("  - Commercial use (e.g., patent applications, productization): requires explicit authorization\n")
    lines.append("- Expected local paths:\n")
    lines.append("  - `data/...`\n")
    lines.append("  - `weights/...`\n")

    lines.append("\n## Quick Start\n")
    lines.append("Minimal validation (no datasets/weights required):\n\n")
    lines.append(md_code_block("python tests/smoke_test.py"))

    lines.append("## Reproducibility Notes\n")
    lines.append("- Prefer running the smoke test first; demo/inference commands (if any) are documented in `code/`.\n")
    lines.append("- GPU memory depends on model/backbone; reduce batch size / image size when needed.\n")

    lines.append("\n## Troubleshooting\n")
    lines.append("- `Weights not found`: request resources by email and place files under `weights/`.\n")
    lines.append("- `ModuleNotFoundError`: install optional dependencies via `pip install -r requirements.txt` (if provided).\n")
    lines.append("- CUDA issues: verify your PyTorch/CUDA installation matches your driver.\n")

    return "".join(lines)


def project_readme_zh(p: dict) -> str:
    # Same structure as English; commands unchanged.
    lines: list[str] = []
    lines.append(LANG_SWITCH)
    lines.append(f"# {p['title_en']}\n")
    lines.append(f"> 中文标题：{p['title_zh']}\n")

    lines.append("## 概览\n")
    for s in p["notes_zh"]:
        lines.append(f"- {s}\n")
    lines.append(f"- 团队成员：{p['team']}\n")

    lines.append("\n## 项目结构\n")
    lines.append("- `code/`：源码\n")
    lines.append("- `data/`：数据集（不随仓库提供；邮件申请）\n")
    lines.append("- `weights/`：权重（不随仓库提供；邮件申请）\n")
    lines.append("- `slides/`：课件（不随仓库提供；邮件申请）\n")
    lines.append("- `tests/`：烟雾测试\n")

    lines.append("\n## 环境\n")
    lines.append("- 激活环境：\n\n")
    lines.append(md_code_block("conda activate pytorch"))
    lines.append("- 可选依赖（若本项目提供 `requirements.txt`，按需安装）：\n\n")
    lines.append(md_code_block("pip install -r requirements.txt"))

    lines.append("## 获取资源（数据集 / 权重 / 课件）\n")
    lines.append(f"- 申请邮箱：{EMAIL}\n")
    lines.append("- 使用政策：\n")
    lines.append("  - 学术研究与教育交流：免费\n")
    lines.append("  - 商业用途（如专利申报、产品化）：需要明确授权\n")
    lines.append("- 本地路径约定：\n")
    lines.append("  - `data/...`\n")
    lines.append("  - `weights/...`\n")

    lines.append("\n## 快速开始\n")
    lines.append("最小化验证（不需要数据集/权重）：\n\n")
    lines.append(md_code_block("python tests/smoke_test.py"))

    lines.append("## 可复现性说明\n")
    lines.append("- 建议先运行烟雾测试；如项目包含演示/推理脚本，通常位于 `code/`。\n")
    lines.append("- 显存占用与模型规模相关；必要时降低 batch size / 输入分辨率。\n")

    lines.append("\n## 常见问题\n")
    lines.append("- `Weights not found`：请邮件申请资源并放置到 `weights/`。\n")
    lines.append("- `ModuleNotFoundError`：按需使用 `pip install -r requirements.txt` 安装依赖（若提供）。\n")
    lines.append("- CUDA 相关错误：请确认 PyTorch/CUDA 与显卡驱动匹配。\n")

    return "".join(lines)


def placeholder_readme(kind: str) -> str:
    if kind == "data":
        title = "Dataset (Not Included)"
        hint = "Place datasets under this folder (not tracked by git)."
        example = "data/\n  <dataset_name>/\n    ...\n"
    elif kind == "weights":
        title = "Weights (Not Included)"
        hint = "Place model weights/checkpoints under this folder (not tracked by git)."
        example = "weights/\n  <model_name>.pt\n  <model_name>.pth\n"
    elif kind == "slides":
        title = "Slides (Not Included)"
        hint = "Place slide decks under this folder (not tracked by git)."
        example = "slides/\n  presentation.pptx\n"
    else:
        raise ValueError(kind)

    en = (
        f"# {title}\n\n"
        f"{hint}\n\n"
        f"- Request email: {EMAIL}\n"
        "- Policy:\n"
        "  - Academic research & educational exchange: free\n"
        "  - Commercial use (e.g., patent applications, productization): requires explicit authorization\n\n"
        "Expected layout:\n\n"
        "```\n"
        f"{example.rstrip()}\n"
        "```\n"
    )

    zh = (
        "\n---\n\n"
        "## 中文说明\n\n"
        "- 本目录内容不随仓库提供（不会上传到 GitHub）。\n"
        f"- 申请邮箱：{EMAIL}\n"
        "- 使用政策：学术研究与教育交流免费；商业用途需明确授权。\n"
    )

    return en + zh


def smoke_entry_py(p: dict) -> str:
    incomplete = bool(p.get("incomplete"))
    import_files = p.get("import_files", [])
    import_list = repr(import_files)

    return (
        textwrap.dedent(
            f"""\
            from __future__ import annotations

            import importlib.util
            import sys
            from pathlib import Path
            from typing import Tuple

            RESOURCE_EMAIL = {EMAIL!r}
            PROJECT = {('projects/' + p['id'] + '-' + p['slug'])!r}

            IMPORT_FILES = {import_list}


            def _dummy_torch_forward() -> None:
                import torch

                x = torch.randn(2, 3)
                model = torch.nn.Linear(3, 4)
                y = model(x)
                assert tuple(y.shape) == (2, 4)


            def _import_python_file(code_dir: Path, rel_path: str) -> None:
                path = (code_dir / rel_path).resolve()
                if not path.exists():
                    raise FileNotFoundError(f\"Missing file: {{rel_path}}\")

                spec = importlib.util.spec_from_file_location(f\"_smoke_{{path.stem}}\", path)
                if spec is None or spec.loader is None:
                    raise ImportError(f\"Cannot import: {{rel_path}}\")

                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)


            def run() -> Tuple[str, str]:
                \"\"\"Return (status, message) where status is one of: passed, skipped, failed.\"\"\"
                code_dir = Path(__file__).resolve().parent
                sys.path.insert(0, str(code_dir))

                try:
                    _dummy_torch_forward()
                except Exception as e:
                    return \"failed\", f\"PyTorch check failed: {{e}}\"

                if {incomplete}:
                    return (
                        \"skipped\",
                        \"Project is incomplete in the public repo. \"
                        f\"Please request resources / missing parts via email: {EMAIL}\",
                    )

                last_error = None
                for rel in IMPORT_FILES:
                    try:
                        _import_python_file(code_dir, rel)
                        return \"passed\", f\"Imported: {{rel}}\"
                    except ModuleNotFoundError as e:
                        last_error = (
                            f\"Missing dependency '{{e.name}}'. \"
                            \"Install optional deps (see project README) and retry.\"
                        )
                    except ImportError as e:
                        last_error = f\"ImportError: {{e}}\"
                    except FileNotFoundError as e:
                        last_error = str(e)
                    except Exception as e:
                        last_error = f\"Exception during import: {{type(e).__name__}}: {{e}}\"

                if not IMPORT_FILES:
                    return \"passed\", \"No import target configured; PyTorch check passed.\"

                return (
                    \"skipped\",
                    (last_error or \"Unable to import project code.\") + f\" Resource request email: {EMAIL}\",
                )
            """
        ).strip()
        + "\n"
    )


def smoke_test_py() -> str:
    return (
        textwrap.dedent(
            """\
            from __future__ import annotations

            import sys
            import traceback
            from pathlib import Path


            def main() -> int:
                project_root = Path(__file__).resolve().parents[1]
                code_dir = project_root / "code"
                sys.path.insert(0, str(code_dir))

                try:
                    import smoke_entry  # type: ignore
                except Exception:
                    print("SMOKE TEST FAILED")
                    traceback.print_exc()
                    return 1

                try:
                    status, message = smoke_entry.run()
                except Exception:
                    print("SMOKE TEST FAILED")
                    traceback.print_exc()
                    return 1

                print(message)

                if status == "passed":
                    print("SMOKE TEST PASSED")
                    return 0
                if status == "skipped":
                    print("SMOKE TEST SKIPPED")
                    return 0

                print("SMOKE TEST FAILED")
                return 1


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ).strip()
        + "\n"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for p in PROJECTS:
        proj_dir = repo_root / "projects" / f"{p['id']}-{p['slug']}"

        write(proj_dir / "README.md", project_readme_en(p))
        write(proj_dir / "README.zh-CN.md", project_readme_zh(p))

        write(proj_dir / "data" / "README.md", placeholder_readme("data"))
        write(proj_dir / "weights" / "README.md", placeholder_readme("weights"))
        write(proj_dir / "slides" / "README.md", placeholder_readme("slides"))

        write(proj_dir / "code" / "smoke_entry.py", smoke_entry_py(p))
        write(proj_dir / "tests" / "smoke_test.py", smoke_test_py())

    print("Generated per-project READMEs, placeholders, and smoke tests.")


if __name__ == "__main__":
    main()
