from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any

import cv2

from .blink_detector import BlinkDetector


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

    def process_frames(
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

        # Print final JSON to terminal for review
        print(json.dumps({
            "type": "blink_processing_summary",
            "summary": summary,
        }, ensure_ascii=False))

        return {
            "timeline": timeline,
            "summary": summary,
        }
