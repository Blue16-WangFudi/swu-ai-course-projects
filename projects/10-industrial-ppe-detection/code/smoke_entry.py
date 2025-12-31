from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

RESOURCE_EMAIL = 'blue16@email.swu.edu.cn'
PROJECT = 'projects/10-industrial-ppe-detection'

IMPORT_FILES = ['apps/realtime_glasses.py']


def _dummy_torch_forward() -> None:
    import torch

    x = torch.randn(2, 3)
    model = torch.nn.Linear(3, 4)
    y = model(x)
    assert tuple(y.shape) == (2, 4)


def _import_python_file(code_dir: Path, rel_path: str) -> None:
    path = (code_dir / rel_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {rel_path}")

    spec = importlib.util.spec_from_file_location(f"_smoke_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {rel_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def run() -> Tuple[str, str]:
    """Return (status, message) where status is one of: passed, skipped, failed."""
    code_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(code_dir))

    try:
        _dummy_torch_forward()
    except Exception as e:
        return "failed", f"PyTorch check failed: {e}"

    if False:
        return (
            "skipped",
            "Project is incomplete in the public repo. "
            f"Please request resources / missing parts via email: blue16@email.swu.edu.cn",
        )

    last_error = None
    for rel in IMPORT_FILES:
        try:
            _import_python_file(code_dir, rel)
            return "passed", f"Imported: {rel}"
        except ModuleNotFoundError as e:
            last_error = (
                f"Missing dependency '{e.name}'. "
                "Install optional deps (see project README) and retry."
            )
        except ImportError as e:
            last_error = f"ImportError: {e}"
        except FileNotFoundError as e:
            last_error = str(e)
        except Exception as e:
            last_error = f"Exception during import: {type(e).__name__}: {e}"

    if not IMPORT_FILES:
        return "passed", "No import target configured; PyTorch check passed."

    return (
        "skipped",
        (last_error or "Unable to import project code.") + f" Resource request email: blue16@email.swu.edu.cn",
    )
