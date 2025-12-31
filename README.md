<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

# swu-ai-course-projects

Standardized, reproducible course projects (16 team projects) from teacher Wu Pengcheng (吴鹏程).

- Mentor page: https://ai.swu.edu.cn/info/1071/2528.htm
- Resources request (datasets / weights / slides): blue16@email.swu.edu.cn

## Overview

This repository collects open-sourced academic course projects covering computer vision, speech, NLP, and intelligent systems.

To make the projects easier to reproduce, this repo standardizes:

- Consistent project layout under `projects/`
- Bilingual READMEs (English + Simplified Chinese) with one-click switching
- Robust `.gitignore` rules to prevent committing large binaries
- Minimal smoke tests (prefer demo/inference; training is documented but not required)

## Repository Structure

```
projects/
  01-soccer-offside-yolov11/
  ...
  16-autonomous-obstacle-avoidance/
scripts/
  smoke_all.py
```

Each project follows:

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

## Quick Start

Activate the existing environment:

```
conda activate pytorch
```

If you prefer not to activate (e.g., CI or one-off runs):

```
conda run -n pytorch python scripts/smoke_all.py
```

If you see `ModuleNotFoundError: No module named 'cv2'`, install OpenCV:

```
pip install opencv-python
```

Run all smoke tests:

```
python scripts/smoke_all.py
```

Install optional per-project dependencies (only when needed):

```
pip install -r projects/<NN>-<slug>/requirements.txt
```

Run a single project smoke test:

```
python projects/<NN>-<slug>/tests/smoke_test.py
```

## Projects Index

See `PROJECTS.md`.

## Getting Resources (Dataset / Weights / Slides)

Resources are intentionally not included in this GitHub repository.

- Request email: blue16@email.swu.edu.cn
- Policy:
  - Academic research & educational exchange: free
  - Commercial use (e.g., patent applications, productization): requires explicit authorization

Expected local paths (per project):

- `projects/<NN>-<slug>/data/`
- `projects/<NN>-<slug>/weights/`
- `projects/<NN>-<slug>/slides/`

## Contributing & Issues

- Contributing guide: `CONTRIBUTING.md`
- Please use the GitHub issue templates for bug reports and feature requests.

## License

This repository uses a custom license:

- Free for academic research & educational exchange
- Commercial use requires explicit authorization
- Provided “as is”, without warranty

See `LICENSE` for details.
