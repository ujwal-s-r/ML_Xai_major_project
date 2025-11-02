from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
import mediapipe as mp


@dataclass
class EmotionAnalyzerConfig:
    model_id: str = "dima806/facial_emotions_image_detection"  # HF model id
    top_k: int = 7
    min_detection_confidence: float = 0.5


class EmotionAnalyzer:
    """ViT-based emotion detection with optional face detection crop.

    Uses Hugging Face transformers pipeline for image-classification and
    MediaPipe Face Detection to extract the face region when available.
    """

    def __init__(self, config: EmotionAnalyzerConfig | None = None) -> None:
        self.cfg = config or EmotionAnalyzerConfig()
        # Face detector
        self._mp = mp.solutions
        self._fd = self._mp.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=self.cfg.min_detection_confidence
        )
        # Device
        self.torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # HF model + processor
        self._processor = AutoImageProcessor.from_pretrained(self.cfg.model_id)
        self._model = AutoModelForImageClassification.from_pretrained(self.cfg.model_id)
        self._model.to(self.torch_device)
        self.id2label = self._model.config.id2label if hasattr(self._model.config, "id2label") else {}

    def _first_face_bbox(self, frame) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._fd.process(rgb)
        if not res.detections:
            return None
        det = res.detections[0]
        bbox = det.location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = max(1, int(bbox.width * w))
        bh = max(1, int(bbox.height * h))
        # add margin
        mx = int(0.1 * bw)
        my = int(0.1 * bh)
        x = max(0, x - mx)
        y = max(0, y - my)
        bw = min(w - x, bw + 2 * mx)
        bh = min(h - y, bh + 2 * my)
        return x, y, bw, bh

    def detect_emotion(self, frame) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"success": False}

        bbox = self._first_face_bbox(frame)
        face_img = frame
        face_detected = False
        if bbox is not None:
            x, y, bw, bh = bbox
            face_img = frame[y:y + bh, x:x + bw]
            face_detected = True

        # Run classifier
        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            inputs = self._processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[0]
            emotions: Dict[str, float] = {}
            for idx, p in enumerate(probs.tolist()):
                label = self.id2label.get(idx, str(idx))
                emotions[label] = float(p * 100.0)
            dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
            return {
                "success": True,
                "face_detected": face_detected,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox is not None else None,
                "dominant_emotion": dominant,
                "emotions": emotions,
            }
        except Exception as e:
            # Ensure we don't reference undefined locals; return bbox if available
            return {
                "success": False,
                "face_detected": face_detected,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox is not None else None,
                "error": str(e),
            }

    def visualize_emotion_detection(self, frame, result: Dict[str, Any]):
        img = frame.copy()
        h, w = img.shape[:2]
        if result is None:
            return img
        bbox = result.get("bbox")
        if bbox:
            x, y, bw, bh = bbox
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            bw = max(1, min(w - x, bw))
            bh = max(1, min(h - y, bh))
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        dom = result.get("dominant_emotion")
        if dom:
            cv2.putText(img, f"{dom}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw top emotions as bars
        emotions: Dict[str, float] = result.get("emotions", {}) or {}
        y0 = 60
        for i, (k, v) in enumerate(sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]):
            y = y0 + i * 22
            cv2.putText(img, f"{k}: {v:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # bar
            bar_len = int(2.5 * v)
            cv2.rectangle(img, (160, y - 10), (160 + bar_len, y - 4), (0, 255, 0), -1)
        return img
