<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

# Badminton Shuttlecock Counting System Based on YOLOv8-P2
> Chinese title: 基于 YOLOv8-P2 的羽毛球颠球自动计数系统
## Overview
- Count shuttlecock juggling repetitions from a single-camera video.
- Uses detection/tracking (YOLOv8-P2 family) to identify the shuttlecock over time.
- Inputs are videos; outputs are on-screen overlays and a final count.
- Team: 吴杰, 李思源

## Project Structure
- `code/`: source code
- `data/`: dataset (not included; request by email)
- `weights/`: weights (not included; request by email)
- `slides/`: slides (not included; request by email)
- `tests/`: smoke tests

## Environment
- Activate environment:

```
conda activate pytorch
```
- Optional extra deps (if this project provides `requirements.txt`):

```
pip install -r requirements.txt
```
## Getting Resources (Dataset / Weights / Slides)
- Email: blue16@email.swu.edu.cn
- Policy:
  - Academic research & educational exchange: free
  - Commercial use (e.g., patent applications, productization): requires explicit authorization
- Expected local paths:
  - `data/...`
  - `weights/...`

## Quick Start
Minimal validation (no datasets/weights required):

```
python tests/smoke_test.py
```
## Reproducibility Notes
- Prefer running the smoke test first; demo/inference commands (if any) are documented in `code/`.
- GPU memory depends on model/backbone; reduce batch size / image size when needed.

## Troubleshooting
- `Weights not found`: request resources by email and place files under `weights/`.
- `ModuleNotFoundError`: install optional dependencies via `pip install -r requirements.txt` (if provided).
- CUDA issues: verify your PyTorch/CUDA installation matches your driver.
