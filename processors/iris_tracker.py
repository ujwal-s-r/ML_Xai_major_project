from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe FaceMesh landmark indices
# Iris landmarks (approx): left iris 468-472, right iris 473-477. We'll use first 5 per iris.
LEFT_IRIS_LMS = [468, 469, 470, 471, 472]
RIGHT_IRIS_LMS = [473, 474, 475, 476, 477]

# Eye corner landmarks for normalization (approx)
LEFT_EYE_CORNERS = (33, 133)      # left eye outer and inner corners
RIGHT_EYE_CORNERS = (362, 263)    # right eye outer and inner corners


@dataclass
class IrisTrackerConfig:
    calib_frames: int = 60  # ~2s at 30fps
    ema_alpha: float = 0.4  # optional smoothing on pupil size


class IrisTracker:
    """MediaPipe Iris tracking and pupil dilation estimation.

    Computes per-eye pupil size (normalized by eye width) and dilation ratio vs baseline.
    """

    # Expose for tests' drawing convenience
    LEFT_IRIS_LANDMARKS = LEFT_IRIS_LMS
    RIGHT_IRIS_LANDMARKS = RIGHT_IRIS_LMS

    def __init__(self, config: IrisTrackerConfig | None = None) -> None:
        self.cfg = config or IrisTrackerConfig()
        self._mp = mp.solutions.face_mesh
        self._fm = self._mp.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # State for baseline and metrics
        self._calib_vals: list[float] = []  # avg pupil sizes over frames
        self._baseline: Optional[float] = None
        self._avg_accum: float = 0.0
        self._count: int = 0
        self._min_size: Optional[float] = None
        self._max_size: Optional[float] = None
        self._ema_avg: Optional[float] = None
        self._last_metrics: Dict[str, Any] = {}

    def reset(self) -> None:
        self._calib_vals.clear()
        self._baseline = None
        self._avg_accum = 0.0
        self._count = 0
        self._min_size = None
        self._max_size = None
        self._ema_avg = None
        self._last_metrics = {}

    def close(self) -> None:
        try:
            if hasattr(self, "_fm") and self._fm:
                self._fm.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    @staticmethod
    def _landmark_xy(lm, w: int, h: int) -> Tuple[float, float]:
        return float(lm.x * w), float(lm.y * h)

    @staticmethod
    def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _eye_width(self, lms, w: int, h: int, corners: Tuple[int, int]) -> Optional[float]:
        try:
            a = self._landmark_xy(lms.landmark[corners[0]], w, h)
            b = self._landmark_xy(lms.landmark[corners[1]], w, h)
            return self._euclid(a, b)
        except Exception:
            return None

    def _iris_center_radius(self, lms, w: int, h: int, iris_idx: list[int]) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        try:
            pts = [self._landmark_xy(lms.landmark[i], w, h) for i in iris_idx]
            if not pts:
                return None, None
            cx = float(np.mean([p[0] for p in pts]))
            cy = float(np.mean([p[1] for p in pts]))
            # Mean radius as average distance to center
            r = float(np.mean([self._euclid((cx, cy), p) for p in pts]))
            return (cx, cy), r
        except Exception:
            return None, None

    def detect_iris(self, frame) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"success": False}

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)

        success = False
        left_size = None
        right_size = None
        avg_size = None
        left_center_norm = None
        right_center_norm = None
        dilation_ratio = None

        try:
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0]

                # Eye widths for normalization
                lw = self._eye_width(lms, w, h, LEFT_EYE_CORNERS)
                rw = self._eye_width(lms, w, h, RIGHT_EYE_CORNERS)

                # Iris centers and radii (pixels)
                l_center, l_r = self._iris_center_radius(lms, w, h, LEFT_IRIS_LMS)
                r_center, r_r = self._iris_center_radius(lms, w, h, RIGHT_IRIS_LMS)

                if lw and rw and l_center and r_center and l_r and r_r:
                    # Normalize sizes by respective eye width to be scale-invariant
                    left_size = float(l_r / lw)
                    right_size = float(r_r / rw)
                    avg_size = float((left_size + right_size) / 2.0)
                    left_center_norm = (l_center[0] / w, l_center[1] / h)
                    right_center_norm = (r_center[0] / w, r_center[1] / h)
                    success = True

        except Exception:
            success = False

        # Update baseline and statistics if we have a valid measurement
        if success and avg_size is not None:
            # EMA smoothing over avg_size
            self._ema_avg = avg_size if self._ema_avg is None else (
                self.cfg.ema_alpha * avg_size + (1 - self.cfg.ema_alpha) * self._ema_avg
            )
            smoothed = float(self._ema_avg)

            if self._baseline is None and len(self._calib_vals) < self.cfg.calib_frames:
                self._calib_vals.append(smoothed)
                if len(self._calib_vals) >= self.cfg.calib_frames:
                    self._baseline = float(np.mean(self._calib_vals))
            # Stats
            self._avg_accum += smoothed
            self._count += 1
            self._min_size = smoothed if self._min_size is None else min(self._min_size, smoothed)
            self._max_size = smoothed if self._max_size is None else max(self._max_size, smoothed)

            if self._baseline:
                dilation_ratio = float((smoothed - self._baseline) / self._baseline) if self._baseline > 0 else 0.0

        metrics = {
            "success": bool(success),
            "landmarks": res.multi_face_landmarks[0] if res and res.multi_face_landmarks else None,
            "left_pupil_size": left_size,
            "right_pupil_size": right_size,
            "avg_pupil_size": avg_size,
            "left_iris_center": left_center_norm,
            "right_iris_center": right_center_norm,
            "pupil_dilation_ratio": dilation_ratio,
            "baseline_locked": bool(self._baseline is not None),
            "baseline_pupil_size": float(self._baseline) if self._baseline is not None else None,
        }
        self._last_metrics = metrics
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        avg = (self._avg_accum / self._count) if self._count > 0 else 0.0
        return {
            "avg_pupil_size": float(avg),
            "max_pupil_size": float(self._max_size) if self._max_size is not None else 0.0,
            "min_pupil_size": float(self._min_size) if self._min_size is not None else 0.0,
            "pupil_dilation_delta": float(avg - (self._baseline or 0.0)),
            "baseline_pupil_size": float(self._baseline) if self._baseline is not None else 0.0,
            "frames_counted": int(self._count),
        }
