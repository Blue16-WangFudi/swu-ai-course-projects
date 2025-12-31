# Contributing

Thanks for improving `swu-ai-course-projects`! This repository focuses on reproducibility and clean project structure.

## Issues

Please use the GitHub issue templates:

- Bug report: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Feature request: `.github/ISSUE_TEMPLATE/feature_request.yml`

When reporting issues, include:

- Project folder (e.g., `projects/09-dialectsense-wavlm`)
- OS + GPU + CUDA/PyTorch version (if relevant)
- Full command and the full traceback/log

## Pull Requests

### Workflow

1. Fork the repository and create a feature branch.
2. Make focused changes with minimal unrelated formatting/renames.
3. Run smoke tests before opening a PR:

```
conda activate pytorch
python scripts/smoke_all.py
```

4. Open a PR describing what changed and how it was validated.

### Branch naming

Use a simple, consistent pattern:

- `docs/<short-topic>`
- `fix/<short-topic>`
- `feat/<short-topic>`
- `refactor/<short-topic>`

Examples:

- `docs/root-readme`
- `fix/12-import-paths`

### Commit message conventions

Use a concise conventional style:

- `docs(<scope>): <summary>`
- `fix(<scope>): <summary>`
- `feat(<scope>): <summary>`
- `refactor(<scope>): <summary>`

Examples:

- `docs(root): add bilingual README`
- `fix(08): make smoke test skip without weights`

## Large Files Policy (Important)

Do **not** commit large binaries to GitHub.

- Datasets, model weights, and slides are intentionally excluded from git.
- Put them under each project’s `data/`, `weights/`, and `slides/` folders (these are ignored).
- If you need the resources, request them via email: blue16@email.swu.edu.cn

If you accidentally added a large file to git history in your branch, remove it from the index before opening a PR:

```
git rm --cached <path>
```

## Project Structure & Naming

Please keep the standard layout for each project under `projects/<NN>-<slug>/` and avoid spaces / non-ASCII characters in file paths.

