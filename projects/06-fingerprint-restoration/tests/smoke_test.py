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
