from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmokeResult:
    project: str
    status: str  # PASSED / SKIPPED / FAILED
    note: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def run_one(test_path: Path) -> SmokeResult:
    project_dir = test_path.parents[1]
    project = project_dir.name

    proc = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
    note = _first_nonempty_line(output)

    if proc.returncode != 0:
        return SmokeResult(project=project, status="FAILED", note=note or f"exit code {proc.returncode}")

    if "SMOKE TEST PASSED" in output:
        return SmokeResult(project=project, status="PASSED", note=note)
    if "SMOKE TEST SKIPPED" in output:
        return SmokeResult(project=project, status="SKIPPED", note=note)

    return SmokeResult(project=project, status="FAILED", note=note or "missing status marker")


def main() -> int:
    root = _repo_root()
    projects_dir = root / "projects"

    test_paths = sorted(projects_dir.glob("*/tests/smoke_test.py"))
    if not test_paths:
        print("No smoke tests found under projects/*/tests/smoke_test.py")
        return 1

    results: list[SmokeResult] = []
    failed_details: dict[str, str] = {}

    for test_path in test_paths:
        res = run_one(test_path)
        results.append(res)

        if res.status == "FAILED":
            # rerun to capture full output for debugging
            proc = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=str(test_path.parents[1]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            failed_details[res.project] = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")

    passed = sum(1 for r in results if r.status == "PASSED")
    skipped = sum(1 for r in results if r.status == "SKIPPED")
    failed = sum(1 for r in results if r.status == "FAILED")

    print("\nSmoke test summary")
    print(f"Total: {len(results)}  PASSED: {passed}  SKIPPED: {skipped}  FAILED: {failed}\n")

    width = max(len(r.project) for r in results)
    for r in results:
        note = (r.note or "").replace("\t", " ")
        if len(note) > 120:
            note = note[:117] + "..."
        print(f"{r.project.ljust(width)}  {r.status.ljust(7)}  {note}")

    if failed:
        print("\nFAILED details")
        for project, out in failed_details.items():
            print(f"\n--- {project} ---")
            print(out.rstrip())
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

