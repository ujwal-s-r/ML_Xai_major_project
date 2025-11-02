from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any

import cv2

from .blink_detector import BlinkDetector
from .l2cs_gaze_tracker import L2CSGazeTracker


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
        try:
            self.gaze = L2CSGazeTracker()
        except Exception as e:
            # Defer error until gaze stage; blink can still run
            self.gaze = None  # type: ignore
            self._gaze_error = e
        else:
            self._gaze_error = None

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
        if self.gaze is None:
            raise RuntimeError(f"Gaze tracker not available: {self._gaze_error}")
        frames_dir = Path(frames_dir)
        image_paths: List[Path] = sorted(p for p in frames_dir.glob("*.jpg"))
        total = len(image_paths)

        timeline: List[Dict[str, Any]] = []
        counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
        for i, img_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(img_path))
            result = self.gaze.detect_gaze(frame)
            if result.get("success"):
                dirc = result.get("direction") or "CENTER"
                counts[dirc] = counts.get(dirc, 0) + 1
            timeline.append(
                {
                    "frame_index": i,
                    "file": img_path.name,
                    "success": bool(result.get("success")),
                    "pitch": result.get("pitch"),
                    "yaw": result.get("yaw"),
                    "direction": result.get("direction"),
                    "vertical": result.get("vertical"),
                }
            )
            if on_progress:
                on_progress(Progress(stage="gaze", processed=i, total=total))

        total_success = sum(1 for t in timeline if t.get("success"))
        summary = {
            "frames_processed": total,
            "frames_with_face": total_success,
            "gaze_distribution": counts,
        }
        return {"timeline": timeline, "summary": summary}

    def process_pipeline(self, frames_dir: Path, on_progress: Callable[[Progress], None] | None = None) -> Dict[str, Any]:
        blink = self.process_frames_blink(frames_dir, on_progress)
        gaze = self.process_frames_gaze(frames_dir, on_progress)
        # Print combined JSON for terminal review
        print(json.dumps({
            "type": "processing_summary",
            "blink": blink["summary"],
            "gaze": gaze["summary"],
        }, ensure_ascii=False))
        return {"blink": blink, "gaze": gaze}
