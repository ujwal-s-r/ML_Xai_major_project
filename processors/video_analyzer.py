from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any

import cv2

from .blink_detector import BlinkDetector
from .gaze_percentile_estimator import GazePercentileEstimator, GazePercentileConfig
from .iris_tracker import IrisTracker
from .emotion_analyzer import EmotionAnalyzer


@dataclass
class Progress:
    stage: str
    processed: int
    total: int


class VideoAnalyzer:
    """Main pipeline to process extracted frames sequentially.

    For now it only runs BlinkDetector over each frame.
    """

    def __init__(self):
        self.blink = BlinkDetector()
        # Initialize new horizontal-only gaze estimator (percentile method)
        try:
            self.gaze_pct = GazePercentileEstimator(GazePercentileConfig())
            self._gaze_error = None
        except Exception as e:
            self.gaze_pct = None  # type: ignore
            self._gaze_error = e
        # Iris tracker for pupil dilation
        try:
            self.iris = IrisTracker()
            self._iris_error = None
        except Exception as e:
            self.iris = None  # type: ignore
            self._iris_error = e
        # Emotion analyzer
        try:
            self.emotion = EmotionAnalyzer()
            self._emotion_error = None
        except Exception as e:
            self.emotion = None  # type: ignore
            self._emotion_error = e

    def process_frames_blink(
        self,
        frames_dir: Path,
        on_progress: Callable[[Progress], None] | None = None,
    ) -> Dict[str, Any]:
        frames_dir = Path(frames_dir)
        image_paths: List[Path] = sorted(
            p for p in frames_dir.glob("*.jpg")
        )
        total = len(image_paths)

        timeline: List[Dict[str, Any]] = []
        for i, img_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(img_path))
            result = self.blink.detect_blink(frame)
            timeline.append(
                {
                    "frame_index": i,
                    "file": img_path.name,
                    "success": bool(result.get("success")),
                    "avg_ear": result.get("avg_ear"),
                    "blink_count": result.get("blink_count"),
                }
            )
            if on_progress:
                on_progress(Progress(stage="blink", processed=i, total=total))

        # Build summary
        avg_ear_values = [t["avg_ear"] for t in timeline if t.get("avg_ear") is not None]
        summary = {
            "frames_processed": total,
            "total_blinks": int(self.blink.blink_counter),
            "avg_ear": float(sum(avg_ear_values) / len(avg_ear_values)) if avg_ear_values else None,
        }

        return {"timeline": timeline, "summary": summary}

    def process_frames_gaze(
        self,
        frames_dir: Path,
        on_progress: Callable[[Progress], None] | None = None,
    ) -> Dict[str, Any]:
        if self.gaze_pct is None:
            raise RuntimeError(f"Gaze tracker not available: {self._gaze_error}")
        frames_dir = Path(frames_dir)
        image_paths: List[Path] = sorted(p for p in frames_dir.glob("*.jpg"))
        total = len(image_paths)

        # Prepare per-frame JSONL output next to frames directory
        out_dir = frames_dir
        per_frame_path = out_dir / "gaze_frames.jsonl"
        # Reset estimator state
        self.gaze_pct.reset()

        timeline: List[Dict[str, Any]] = []
        counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
        frames_with_face = 0

        with open(per_frame_path, "w", encoding="utf-8") as pf:
            for i, img_path in enumerate(image_paths, start=1):
                frame = cv2.imread(str(img_path))
                metrics = self.gaze_pct.analyze_frame(frame)

                # Update counts
                label = metrics.get("label") or "N/A"
                if metrics.get("face_detected"):
                    frames_with_face += 1
                if label in counts:
                    counts[label] += 1

                # Minimal timeline entry for debugging
                timeline.append({
                    "frame_index": i,
                    "file": img_path.name,
                    "success": bool(metrics.get("success")),
                    "label": label,
                    "confidence": metrics.get("confidence"),
                })

                # Persist per-frame JSONL for future visualization
                rec = {
                    "frame_index": i,
                    "file": img_path.name,
                    "face_detected": bool(metrics.get("face_detected")),
                    "eyes_detected": bool(metrics.get("eyes_detected")),
                    "label": label,
                    "confidence": metrics.get("confidence"),
                    "avg_left_perc": metrics.get("avg_left_perc"),
                    "avg_right_perc": metrics.get("avg_right_perc"),
                    "left_eye_roi_abs": metrics.get("left_eye_roi_abs"),
                    "right_eye_roi_abs": metrics.get("right_eye_roi_abs"),
                    "left_pupil_abs": metrics.get("left_pupil_abs"),
                    "right_pupil_abs": metrics.get("right_pupil_abs"),
                    "debug": metrics.get("debug"),
                }
                pf.write(json.dumps(rec) + "\n")

                if on_progress:
                    on_progress(Progress(stage="gaze", processed=i, total=total))

        summary = {
            "frames_processed": total,
            "frames_with_face": frames_with_face,
            "gaze_distribution": counts,
            "per_frame_path": str(per_frame_path),
        }
        return {"timeline": timeline, "summary": summary}

    def process_frames_pupil(
        self,
        frames_dir: Path,
        on_progress: Callable[[Progress], None] | None = None,
    ) -> Dict[str, Any]:
        if self.iris is None:
            raise RuntimeError(f"Iris tracker not available: {self._iris_error}")
        frames_dir = Path(frames_dir)
        image_paths: List[Path] = sorted(p for p in frames_dir.glob("*.jpg"))
        total = len(image_paths)

        per_frame_path = frames_dir / "pupil_frames.jsonl"
        # reset metrics
        self.iris.reset()

        frames_with_iris = 0
        timeline: List[Dict[str, Any]] = []

        with open(per_frame_path, "w", encoding="utf-8") as pf:
            for i, img_path in enumerate(image_paths, start=1):
                frame = cv2.imread(str(img_path))
                metrics = self.iris.detect_iris(frame)
                success = bool(metrics.get("success"))
                if success:
                    frames_with_iris += 1

                # Minimal timeline
                timeline.append({
                    "frame_index": i,
                    "file": img_path.name,
                    "success": success,
                    "avg_pupil_size": metrics.get("avg_pupil_size"),
                    "pupil_dilation_ratio": metrics.get("pupil_dilation_ratio"),
                })

                # Persist per-frame JSONL
                rec = {
                    "frame_index": i,
                    "file": img_path.name,
                    "success": success,
                    "left_pupil_size": metrics.get("left_pupil_size"),
                    "right_pupil_size": metrics.get("right_pupil_size"),
                    "avg_pupil_size": metrics.get("avg_pupil_size"),
                    "left_iris_center": metrics.get("left_iris_center"),
                    "right_iris_center": metrics.get("right_iris_center"),
                    "pupil_dilation_ratio": metrics.get("pupil_dilation_ratio"),
                    "baseline_locked": metrics.get("baseline_locked"),
                    "baseline_pupil_size": metrics.get("baseline_pupil_size"),
                }
                pf.write(json.dumps(rec) + "\n")

                if on_progress:
                    on_progress(Progress(stage="pupil", processed=i, total=total))

        # Summary stats
        stats = self.iris.get_metrics()
        summary = {
            "frames_processed": total,
            "frames_with_iris": frames_with_iris,
            "avg_pupil_size": stats.get("avg_pupil_size"),
            "min_pupil_size": stats.get("min_pupil_size"),
            "max_pupil_size": stats.get("max_pupil_size"),
            "baseline_pupil_size": stats.get("baseline_pupil_size"),
            "per_frame_path": str(per_frame_path),
        }
        return {"timeline": timeline, "summary": summary}

    def process_frames_emotion(
        self,
        frames_dir: Path,
        on_progress: Callable[[Progress], None] | None = None,
    ) -> Dict[str, Any]:
        if self.emotion is None:
            raise RuntimeError(f"Emotion analyzer not available: {self._emotion_error}")
        frames_dir = Path(frames_dir)
        image_paths: List[Path] = sorted(p for p in frames_dir.glob("*.jpg"))
        total = len(image_paths)

        per_frame_path = frames_dir / "emotion_frames.jsonl"
        counts: Dict[str, int] = {}
        frames_with_face = 0
        frames_classified = 0
        error_count = 0
        last_error = None
        timeline: List[Dict[str, Any]] = []

        with open(per_frame_path, "w", encoding="utf-8") as pf:
            for i, img_path in enumerate(image_paths, start=1):
                frame = cv2.imread(str(img_path))
                result = self.emotion.detect_emotion(frame)
                success = bool(result.get("success"))
                if result.get("face_detected"):
                    frames_with_face += 1
                dom = result.get("dominant_emotion")
                if dom:
                    counts[dom] = counts.get(dom, 0) + 1
                if success:
                    frames_classified += 1
                if not success and result.get("error"):
                    error_count += 1
                    last_error = result.get("error")

                # minimal timeline
                timeline.append({
                    "frame_index": i,
                    "file": img_path.name,
                    "success": success,
                    "dominant_emotion": dom,
                })

                # per-frame JSONL
                rec = {
                    "frame_index": i,
                    "file": img_path.name,
                    "success": success,
                    "face_detected": bool(result.get("face_detected")),
                    "bbox": result.get("bbox"),
                    "dominant_emotion": dom,
                    "emotions": result.get("emotions"),
                    "error": result.get("error"),
                }
                pf.write(json.dumps(rec) + "\n")

                if on_progress:
                    on_progress(Progress(stage="emotion", processed=i, total=total))

        summary = {
            "frames_processed": total,
            "frames_with_face": frames_with_face,
            "frames_classified": frames_classified,
            "errors": error_count,
            "last_error": last_error,
            "dominant_counts": counts,
            "per_frame_path": str(per_frame_path),
        }
        return {"timeline": timeline, "summary": summary}

    def process_pipeline(self, frames_dir: Path, on_progress: Callable[[Progress], None] | None = None) -> Dict[str, Any]:
        blink = self.process_frames_blink(frames_dir, on_progress)
        gaze = self.process_frames_gaze(frames_dir, on_progress)
        pupil = self.process_frames_pupil(frames_dir, on_progress)
        emotion = self.process_frames_emotion(frames_dir, on_progress)
        # Print combined JSON for terminal review
        print(json.dumps({
            "type": "processing_summary",
            "blink": blink["summary"],
            "gaze": gaze["summary"],
            "pupil": pupil["summary"],
            "emotion": emotion["summary"],
        }, ensure_ascii=False))
        return {"blink": blink, "gaze": gaze, "pupil": pupil, "emotion": emotion}
