from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe FaceMesh landmark indices
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
LEFT_PUPIL_LANDMARK = 473
RIGHT_PUPIL_LANDMARK = 468


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _bbox_from_landmarks(landmarks, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
    if not landmarks:
        return None
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    if not xs or not ys:
        return None
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)
    x_min = _clamp(x_min, 0, max(0, frame_w - 1))
    y_min = _clamp(y_min, 0, max(0, frame_h - 1))
    w = _clamp(w, 1, max(1, frame_w - x_min))
    h = _clamp(h, 1, max(1, frame_h - y_min))
    return x_min, y_min, w, h


def _process_eye(roi_gray, eye_rect, fallback_thresh: int = 50):
    """Return ((left_perc, right_perc), (cx, cy), threshold_mode, used_thresh) for one eye.
    eye_rect is (ex, ey, ew, eh) relative to roi_gray.
    """
    ex, ey, ew, eh = eye_rect
    roi_h, roi_w = roi_gray.shape
    ex = _clamp(ex, 0, max(0, roi_w - 1))
    ey = _clamp(ey, 0, max(0, roi_h - 1))
    ew = _clamp(ew, 1, max(1, roi_w - ex))
    eh = _clamp(eh, 1, max(1, roi_h - ey))

    eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
    if eye_roi.size == 0:
        return (50.0, 50.0), (ew // 2, eh // 2), "none", float(fallback_thresh)

    try:
        eye_eq = cv2.equalizeHist(eye_roi)
    except cv2.error:
        eye_eq = eye_roi

    threshold_mode = "otsu"
    used_thresh = -1.0
    try:
        _, thr = cv2.threshold(eye_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except cv2.error:
        threshold_mode = "fixed"
        used_thresh = float(fallback_thresh)
        _, thr = cv2.threshold(eye_eq, fallback_thresh, 255, cv2.THRESH_BINARY_INV)

    mid_x = max(1, ew // 2)
    left_half = thr[:, :mid_x]
    right_half = thr[:, mid_x:]
    left_pixels = cv2.countNonZero(left_half)
    right_pixels = cv2.countNonZero(right_half)
    total = left_pixels + right_pixels
    if total == 0:
        left_perc = 50.0
        right_perc = 50.0
    else:
        left_perc = (left_pixels / float(total)) * 100.0
        right_perc = (right_pixels / float(total)) * 100.0

    m = cv2.moments(thr)
    cx, cy = ew // 2, eh // 2
    if m["m00"] != 0:
        cx = int(m["m10"] / m["m00"]) 
        cy = int(m["m01"] / m["m00"]) 

    return (left_perc, right_perc), (cx, cy), threshold_mode, used_thresh


@dataclass
class GazePercentileConfig:
    fallback_thresh: int = 50
    ema_alpha: float = 0.5
    enter_diff: float = 10.0
    exit_diff: float = 5.0
    calib_frames: int = 30


class GazePercentileEstimator:
    """Horizontal-only gaze estimator using eye percentile method.

    Outputs LEFT/CENTER/RIGHT labels with smoothing and hysteresis.
    """

    def __init__(self, config: GazePercentileConfig | None = None):
        self.cfg = config or GazePercentileConfig()
        self._mp = mp.solutions.face_mesh
        self._fm = self._mp.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # State
        self._calib_vals: list[float] = []  # signal s = (avg_right - avg_left)
        self._baseline_s: float = 0.0
        self._baseline_locked: bool = False
        self._ema_s: Optional[float] = None
        self._state: str = "CENTER"

    def reset(self) -> None:
        self._calib_vals.clear()
        self._baseline_s = 0.0
        self._baseline_locked = False
        self._ema_s = None
        self._state = "CENTER"

    def close(self) -> None:
        try:
            if hasattr(self, "_fm") and self._fm:
                self._fm.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def analyze_frame(self, frame) -> Dict[str, Any]:
        """Analyze a single BGR frame and return metrics.

        Returns keys: success, face_detected, eyes_detected, label, confidence,
        avg_left_perc, avg_right_perc, left_eye_roi_abs, right_eye_roi_abs,
        left_pupil_abs, right_pupil_abs, debug{...}
        """
        if frame is None or frame.size == 0:
            return {"success": False}

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)

        face_detected = False
        eyes_detected = False
        label = "N/A"
        confidence = 0.0
        avg_left = 50.0
        avg_right = 50.0
        left_eye_abs = None
        right_eye_abs = None
        left_pupil_abs = None
        right_pupil_abs = None
        threshold_mode_agg = "none"
        used_thresh_agg = -1.0
        ema_signal_out = None

        if res.multi_face_landmarks:
            face_detected = True
            lms = res.multi_face_landmarks[0]
            all_lms = [lm for lm in lms.landmark]
            face_box = _bbox_from_landmarks(all_lms, w, h)
            if face_box:
                fx, fy, fw, fh = face_box
                roi_gray = gray[fy:fy + fh, fx:fx + fh]

                # Eye boxes (absolute)
                left_eye_lm = [lms.landmark[i] for i in LEFT_EYE_LANDMARKS]
                right_eye_lm = [lms.landmark[i] for i in RIGHT_EYE_LANDMARKS]
                left_eye_box_abs = _bbox_from_landmarks(left_eye_lm, w, h)
                right_eye_box_abs = _bbox_from_landmarks(right_eye_lm, w, h)

                # Pupil landmarks (absolute)
                try:
                    lp_lm = lms.landmark[LEFT_PUPIL_LANDMARK]
                    rp_lm = lms.landmark[RIGHT_PUPIL_LANDMARK]
                    left_pupil_abs = [int(lp_lm.x * w), int(lp_lm.y * h)]
                    right_pupil_abs = [int(rp_lm.x * w), int(rp_lm.y * h)]
                except Exception:
                    pass

                if left_eye_box_abs and right_eye_box_abs:
                    lx_abs, ly_abs, lw, lh = left_eye_box_abs
                    rx_abs, ry_abs, rw, rh = right_eye_box_abs
                    # Relative to face ROI
                    lx_rel, ly_rel = lx_abs - fx, ly_abs - fy
                    rx_rel, ry_rel = rx_abs - fx, ry_abs - fy
                    lx_rel = _clamp(lx_rel, 0, max(0, fw - 1))
                    ly_rel = _clamp(ly_rel, 0, max(0, fh - 1))
                    rx_rel = _clamp(rx_rel, 0, max(0, fw - 1))
                    ry_rel = _clamp(ry_rel, 0, max(0, fh - 1))
                    lw = _clamp(lw, 1, max(1, fw - lx_rel))
                    lh = _clamp(lh, 1, max(1, fh - ly_rel))
                    rw = _clamp(rw, 1, max(1, fw - rx_rel))
                    rh = _clamp(rh, 1, max(1, fh - ry_rel))

                    left_perc, _, mode_l, used_l = _process_eye(
                        roi_gray, (lx_rel, ly_rel, lw, lh), self.cfg.fallback_thresh
                    )
                    right_perc, _, mode_r, used_r = _process_eye(
                        roi_gray, (rx_rel, ry_rel, rw, rh), self.cfg.fallback_thresh
                    )

                    eyes_detected = True
                    left_eye_abs = [lx_abs, ly_abs, lw, lh]
                    right_eye_abs = [rx_abs, ry_abs, rw, rh]
                    threshold_mode_agg = "otsu" if (mode_l == "otsu" or mode_r == "otsu") else "fixed"
                    used_thresh_agg = max(float(used_l), float(used_r))

                    # Averages across eyes
                    avg_left = (left_perc[0] + right_perc[0]) / 2.0
                    avg_right = (left_perc[1] + right_perc[1]) / 2.0
                    s = (avg_right - avg_left)  # positive => RIGHT, negative => LEFT

                    # Calibration
                    if not self._baseline_locked:
                        self._calib_vals.append(s)
                        if len(self._calib_vals) >= self.cfg.calib_frames:
                            self._baseline_s = float(np.mean(self._calib_vals))
                            self._baseline_locked = True

                    adj_s = s - (self._baseline_s if self._baseline_locked else 0.0)

                    # EMA smoothing
                    self._ema_s = adj_s if self._ema_s is None else (
                        self.cfg.ema_alpha * adj_s + (1 - self.cfg.ema_alpha) * self._ema_s
                    )
                    ema_signal_out = float(self._ema_s)

                    # Hysteresis classification
                    if self._state == "CENTER":
                        if self._ema_s <= -self.cfg.enter_diff:
                            self._state = "LEFT"
                        elif self._ema_s >= self.cfg.enter_diff:
                            self._state = "RIGHT"
                    elif self._state == "LEFT":
                        if -self.cfg.exit_diff <= self._ema_s <= self.cfg.exit_diff:
                            self._state = "CENTER"
                        elif self._ema_s >= self.cfg.enter_diff:
                            self._state = "RIGHT"
                    elif self._state == "RIGHT":
                        if -self.cfg.exit_diff <= self._ema_s <= self.cfg.exit_diff:
                            self._state = "CENTER"
                        elif self._ema_s <= -self.cfg.enter_diff:
                            self._state = "LEFT"

                    label = self._state
                    confidence = float(abs(adj_s))

        return {
            "success": bool(face_detected and eyes_detected),
            "face_detected": bool(face_detected),
            "eyes_detected": bool(eyes_detected),
            "label": label,
            "confidence": float(confidence),
            "avg_left_perc": float(avg_left),
            "avg_right_perc": float(avg_right),
            "left_eye_roi_abs": left_eye_abs,
            "right_eye_roi_abs": right_eye_abs,
            "left_pupil_abs": left_pupil_abs,
            "right_pupil_abs": right_pupil_abs,
            "debug": {
                "threshold_mode": threshold_mode_agg,
                "used_threshold": float(used_thresh_agg),
                "baseline_locked": bool(self._baseline_locked),
                "baseline_s": float(self._baseline_s),
                "ema_signal": float(ema_signal_out) if ema_signal_out is not None else None,
                "enter_diff": float(self.cfg.enter_diff),
                "exit_diff": float(self.cfg.exit_diff),
                "ema_alpha": float(self.cfg.ema_alpha),
                "calib_frames": int(self.cfg.calib_frames),
            },
        }
