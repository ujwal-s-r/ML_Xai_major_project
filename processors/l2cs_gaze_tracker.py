from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import torch  # noqa: F401
    from l2cs import Pipeline, select_device
except Exception as e:  # pragma: no cover
    Pipeline = None  # type: ignore
    select_device = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass
class GazeResult:
    success: bool
    pitch: float | None = None
    yaw: float | None = None
    direction: str | None = None  # LEFT/CENTER/RIGHT
    vertical: str | None = None   # UP/CENTER/DOWN


class L2CSGazeTracker:
    """Wrapper around L2CS-Net gaze estimation pipeline.

    Exposes a simple detect_gaze(frame) API for the first detected face.
    """

    def __init__(self, weights_path: str | Path | None = None, device: str = "cpu") -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                f"Failed to import L2CS modules: {_IMPORT_ERROR}. "
                "Install with: pip install git+https://github.com/Ahmednull/L2CS-Net.git"
            )
        # Resolve weights
        if weights_path is None:
            # default to repo models directory
            root = Path(__file__).resolve().parents[1]
            weights_path = root / 'models' / 'L2CSNet_gaze360.pkl'
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"L2CS weights not found at {weights_path}. Expected models/L2CSNet_gaze360.pkl"
            )

        dev = select_device(device, batch_size=1)
        self.pipeline = Pipeline(weights=weights_path, arch='ResNet50', device=dev)

    @staticmethod
    def _direction_from_angles(pitch: float, yaw: float) -> tuple[str, str]:
        # Thresholds in degrees similar to test script
        horiz = 'LEFT' if yaw < -15 else 'RIGHT' if yaw > 15 else 'CENTER'
        vert = 'UP' if pitch < -15 else 'DOWN' if pitch > 15 else 'CENTER'
        return vert, horiz

    def detect_gaze(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run gaze estimation. Returns first face result.

        Response keys: success, pitch, yaw, direction (LEFT/CENTER/RIGHT), vertical (UP/CENTER/DOWN)
        """
        if frame is None or frame.size == 0:
            return {"success": False}

        with np.errstate(all='ignore'):
            results = self.pipeline.step(frame)

        if not results:
            return {"success": False}

        # results may be dict-like or object-like
        if isinstance(results, dict):
            pitch_list = results.get('pitch', [])
            yaw_list = results.get('yaw', [])
        else:
            pitch_list = getattr(results, 'pitch', [])
            yaw_list = getattr(results, 'yaw', [])

        if not pitch_list or not yaw_list:
            return {"success": False}

        pitch = float(pitch_list[0])
        yaw = float(yaw_list[0])
        vert, horiz = self._direction_from_angles(pitch, yaw)
        return {
            "success": True,
            "pitch": pitch,
            "yaw": yaw,
            "direction": horiz,
            "vertical": vert,
        }
