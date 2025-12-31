from __future__ import annotations

from typing import Tuple

RESOURCE_EMAIL = "blue16@email.swu.edu.cn"
PROJECT = "projects/11-fatigue-detection-system"


def _dummy_torch_forward() -> None:
    import torch

    x = torch.randn(2, 3)
    model = torch.nn.Linear(3, 4)
    y = model(x)
    assert tuple(y.shape) == (2, 4)


def run() -> Tuple[str, str]:
    """Return (status, message) where status is one of: passed, skipped, failed."""

    try:
        _dummy_torch_forward()
    except Exception as e:
        return "failed", f"PyTorch check failed: {e}"

    try:
        import webcam_fatigue_detection  # noqa: F401
    except ModuleNotFoundError as e:
        return (
            "skipped",
            f"Missing dependency '{e.name}'. Install optional deps (see project README). "
            f"Resource request email: {RESOURCE_EMAIL}",
        )
    except Exception as e:
        return "failed", f"Exception during import: {type(e).__name__}: {e}"

    # Minimal MediaPipe FaceMesh path (no webcam required).
    try:
        import mediapipe as mp
        import numpy as np

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        try:
            dummy_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            _ = face_mesh.process(dummy_rgb)
        finally:
            face_mesh.close()
    except ModuleNotFoundError as e:
        return (
            "skipped",
            f"Missing dependency '{e.name}'. Install optional deps (see project README). "
            f"Resource request email: {RESOURCE_EMAIL}",
        )
    except Exception as e:
        return "failed", f"MediaPipe FaceMesh check failed: {type(e).__name__}: {e}"

    return "passed", "Imported webcam demo safely; MediaPipe FaceMesh ran on a dummy frame."
