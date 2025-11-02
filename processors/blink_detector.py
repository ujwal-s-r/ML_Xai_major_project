from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class BlinkResult:
    success: bool
    left_ear: float | None = None
    right_ear: float | None = None
    avg_ear: float | None = None
    is_blinking: bool | None = None
    blink_count: int | None = None
    landmarks: Any | None = None


class BlinkDetector:
    """Blink detector using MediaPipe Face Mesh and Eye Aspect Ratio (EAR).

    Compatible with tests/test_blink_detection.py API.
    """

    # Using standard FaceMesh indices for eye landmarks (6 points per eye)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]   # p1, p2, p3, p4, p5, p6
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # p1, p2, p3, p4, p5, p6

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.EAR_THRESHOLD: float = 0.20
        self._blink_state: bool = False  # True when currently below threshold
        self.blink_counter: int = 0
        self.ear_history: list[float] = []

    @staticmethod
    def _dist(ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(ax - bx, ay - by)

    def _eye_ear(self, lm, idx, w: int, h: int) -> float:
        # points: p1 - p6
        p1 = lm[idx[0]]; p2 = lm[idx[1]]; p3 = lm[idx[2]]
        p4 = lm[idx[3]]; p5 = lm[idx[4]]; p6 = lm[idx[5]]
        # Convert to pixel coords
        x1, y1 = p1.x * w, p1.y * h
        x2, y2 = p2.x * w, p2.y * h
        x3, y3 = p3.x * w, p3.y * h
        x4, y4 = p4.x * w, p4.y * h
        x5, y5 = p5.x * w, p5.y * h
        x6, y6 = p6.x * w, p6.y * h
        # EAR formula
        vert1 = self._dist(x2, y2, x6, y6)
        vert2 = self._dist(x3, y3, x5, y5)
        horiz = self._dist(x1, y1, x4, y4)
        if horiz <= 1e-6:
            return 0.0
        return (vert1 + vert2) / (2.0 * horiz)

    def detect_blink(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"success": False}

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            # Reset blink state when no face
            self._blink_state = False
            return {"success": False}

        face_landmarks = res.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        left_ear = self._eye_ear(lm, self.LEFT_EYE_INDICES, w, h)
        right_ear = self._eye_ear(lm, self.RIGHT_EYE_INDICES, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > 600:
            self.ear_history.pop(0)

        is_closed = avg_ear < self.EAR_THRESHOLD
        if is_closed and not self._blink_state:
            # Transition: open -> closed
            self._blink_state = True
        elif not is_closed and self._blink_state:
            # Transition: closed -> open  => count one blink
            self.blink_counter += 1
            self._blink_state = False

        return {
            "success": True,
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "avg_ear": float(avg_ear),
            "is_blinking": bool(is_closed),
            "blink_count": int(self.blink_counter),
            "landmarks": face_landmarks,
        }
