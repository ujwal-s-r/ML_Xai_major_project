"""
Batch Video Analyzer - Sequential Processing
Processes all frames sequentially, one model at a time.

Flow:
1. Extract all frames from video
2. Process Emotion (every 10th frame)
3. Process Blink (every 2nd frame)
4. Process Iris (every 2nd frame)
5. Process Gaze (every 3rd frame)
6. Combine results into timeline
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional
import pathlib

from .emotion_analyzer import EmotionAnalyzer
from .blink_detector import BlinkDetector
from .iris_tracker import IrisTracker

# L2CS imports
from l2cs import Pipeline, select_device


class BatchVideoAnalyzer:
    """
    Sequential batch video analyzer that processes frames one model at a time.
    More reliable than parallel processing.
    """
    
    def __init__(self):
        """Initialize all analyzers."""
        print("Initializing BatchVideoAnalyzer...")
        
        # Initialize analyzers
        print("  - Loading Emotion Analyzer...")
        self.emotion_analyzer = EmotionAnalyzer()
        
        print("  - Loading Blink Detector...")
        self.blink_detector = BlinkDetector()
        
        print("  - Loading Iris Tracker...")
        self.iris_tracker = IrisTracker()
        
        print("  - Loading Gaze Estimator (L2CS-Net)...")
        CWD = pathlib.Path.cwd()
        weights_path = CWD / 'models' / 'L2CSNet_gaze360.pkl'
        device = select_device('cpu', batch_size=1)
        self.gaze_pipeline = Pipeline(
            weights=weights_path,
            arch='ResNet50',
            device=device
        )
        
        # Processing state
        self.frames = []
        self.fps = 30.0
        self.timeline = []
        
        print("✓ BatchVideoAnalyzer initialized successfully!")
    
    def set_frames(self, frames: List[np.ndarray], fps: float = 30.0):
        """
        Set the frames to process.
        
        Args:
            frames: List of BGR frames
            fps: Frames per second of video
        """
        self.frames = frames
        self.fps = fps
        self.timeline = []
        print(f"Loaded {len(frames)} frames at {fps} FPS")
    
    def process_all_sequential(self) -> Dict:
        """
        Process all frames sequentially, one model at a time.
        
        Returns:
            Dictionary with timeline and summary
        """
        if not self.frames:
            raise ValueError("No frames loaded. Call set_frames() first.")
        
        total_frames = len(self.frames)
        print(f"\n{'='*60}")
        print(f"Starting Sequential Processing: {total_frames} frames")
        print(f"{'='*60}\n")
        
        # Initialize timeline with frame metadata
        self.timeline = []
        for i in range(total_frames):
            self.timeline.append({
                "timestamp": i / self.fps,
                "frame_number": i,
                "processed": {
                    "emotion": False,
                    "blink": False,
                    "iris": False,
                    "gaze": False
                }
            })
        
        # Process each model sequentially
        self._process_emotion_sequential()
        self._process_blink_sequential()
        self._process_iris_sequential()
        self._process_gaze_sequential()
        
        # Generate summary
        summary = self._generate_summary()
        
        print(f"\n{'='*60}")
        print(f"Sequential Processing Complete!")
        print(f"{'='*60}")
        print(f"Summary Generated:")
        print(f"  Emotion: {summary.get('emotion', {})}")
        print(f"  Blink: {summary.get('blink', {})}")
        print(f"  Pupil: {summary.get('pupil', {})}")
        print(f"  Gaze: {summary.get('gaze', {})}")
        print(f"{'='*60}\n")
        
        return {
            "timeline": self.timeline,
            "summary": summary
        }
    
    def _process_emotion_sequential(self):
        """Process emotion detection (every 10th frame - 3 FPS)."""
        print("\n[1/4] Processing Emotion Analysis (3 FPS)...")
        sample_rate = 10
        frames_to_process = list(range(0, len(self.frames), sample_rate))
        
        for idx, frame_num in enumerate(frames_to_process):
            frame = self.frames[frame_num]
            
            try:
                result = self.emotion_analyzer.detect_emotion(frame)
                
                if result and result.get("success"):
                    dominant = result.get("dominant_emotion", "unknown")
                    emotions_dict = result.get("emotions", {})
                    confidence = emotions_dict.get(dominant, 0.0) if emotions_dict else 0.0
                    
                    self.timeline[frame_num]["emotion"] = {
                        "label": dominant,
                        "confidence": confidence,
                        "scores": emotions_dict
                    }
                    self.timeline[frame_num]["processed"]["emotion"] = True
                else:
                    self.timeline[frame_num]["emotion"] = None
                
            except Exception as e:
                print(f"  ⚠️ Error at frame {frame_num}: {e}")
                import traceback
                traceback.print_exc()
                self.timeline[frame_num]["emotion"] = None
            
            # Progress
            if (idx + 1) % 10 == 0 or idx == len(frames_to_process) - 1:
                progress = (idx + 1) / len(frames_to_process) * 100
                print(f"  Progress: {idx + 1}/{len(frames_to_process)} ({progress:.1f}%)")
        
        print(f"  ✓ Emotion analysis complete!")
    
    def _process_blink_sequential(self):
        """Process blink detection (every 2nd frame - 15 FPS)."""
        print("\n[2/4] Processing Blink Detection (15 FPS)...")
        sample_rate = 2
        frames_to_process = list(range(0, len(self.frames), sample_rate))
        
        # Reset blink detector state
        try:
            self.blink_detector.reset()
            print("  ✓ Blink detector reset")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not reset blink detector: {e}")
        
        for idx, frame_num in enumerate(frames_to_process):
            frame = self.frames[frame_num]
            
            try:
                result = self.blink_detector.detect_blink(frame)
                
                if result and result.get("success"):
                    self.timeline[frame_num]["blink"] = {
                        "ear_left": result.get("left_ear", 0.0),
                        "ear_right": result.get("right_ear", 0.0),
                        "avg_ear": result.get("avg_ear", 0.0),
                        "is_blinking": result.get("is_blinking", False),
                        "cumulative_blinks": result.get("blink_count", 0)
                    }
                    self.timeline[frame_num]["processed"]["blink"] = True
                else:
                    self.timeline[frame_num]["blink"] = None
                    
            except Exception as e:
                print(f"  ⚠️ Error at frame {frame_num}: {e}")
                import traceback
                traceback.print_exc()
                self.timeline[frame_num]["blink"] = None
            
            # Progress
            if (idx + 1) % 20 == 0 or idx == len(frames_to_process) - 1:
                progress = (idx + 1) / len(frames_to_process) * 100
                print(f"  Progress: {idx + 1}/{len(frames_to_process)} ({progress:.1f}%)")
        
        print(f"  ✓ Blink detection complete!")
    
    def _process_iris_sequential(self):
        """Process iris/pupil tracking (every 2nd frame - 15 FPS)."""
        print("\n[3/4] Processing Iris/Pupil Tracking (15 FPS)...")
        sample_rate = 2
        frames_to_process = list(range(0, len(self.frames), sample_rate))
        
        # Reset iris tracker state
        try:
            self.iris_tracker.reset()
            print("  ✓ Iris tracker reset")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not reset iris tracker: {e}")
        
        for idx, frame_num in enumerate(frames_to_process):
            frame = self.frames[frame_num]
            
            try:
                result = self.iris_tracker.detect_iris(frame)
                
                if result and result.get("success"):
                    left_size = result.get("left_pupil_size", 0.0)
                    right_size = result.get("right_pupil_size", 0.0)
                    avg_size = (left_size + right_size) / 2.0 if left_size and right_size else 0.0
                    
                    self.timeline[frame_num]["pupil"] = {
                        "left": left_size,
                        "right": right_size,
                        "avg": avg_size,
                        "dilation_ratio": result.get("pupil_dilation_ratio", 0.0)
                    }
                    self.timeline[frame_num]["processed"]["iris"] = True
                else:
                    self.timeline[frame_num]["pupil"] = None
                    
            except Exception as e:
                print(f"  ⚠️ Error at frame {frame_num}: {e}")
                import traceback
                traceback.print_exc()
                self.timeline[frame_num]["pupil"] = None
            
            # Progress
            if (idx + 1) % 20 == 0 or idx == len(frames_to_process) - 1:
                progress = (idx + 1) / len(frames_to_process) * 100
                print(f"  Progress: {idx + 1}/{len(frames_to_process)} ({progress:.1f}%)")
        
        print(f"  ✓ Iris tracking complete!")
    
    def _process_gaze_sequential(self):
        """Process gaze estimation (every 3rd frame - 10 FPS)."""
        print("\n[4/4] Processing Gaze Estimation (10 FPS)...")
        sample_rate = 3
        frames_to_process = list(range(0, len(self.frames), sample_rate))
        
        for idx, frame_num in enumerate(frames_to_process):
            frame = self.frames[frame_num]
            
            try:
                # L2CS Pipeline returns dictionary with 'pitch' and 'yaw' lists
                results = self.gaze_pipeline.step(frame)
                
                if results and isinstance(results, dict):
                    # Extract pitch and yaw lists
                    pitch_list = results.get('pitch', [])
                    yaw_list = results.get('yaw', [])
                    
                    # Use first detected face
                    if pitch_list and yaw_list and len(pitch_list) > 0 and len(yaw_list) > 0:
                        pitch_val = float(pitch_list[0])
                        yaw_val = float(yaw_list[0])
                        
                        # Determine direction
                        if yaw_val < -15:
                            direction = "left"
                        elif yaw_val > 15:
                            direction = "right"
                        else:
                            direction = "center"
                        
                        self.timeline[frame_num]["gaze"] = {
                            "pitch": pitch_val,
                            "yaw": yaw_val,
                            "direction": direction
                        }
                        self.timeline[frame_num]["processed"]["gaze"] = True
                    else:
                        self.timeline[frame_num]["gaze"] = None
                else:
                    self.timeline[frame_num]["gaze"] = None
                    
            except Exception as e:
                print(f"  ⚠️ Error at frame {frame_num}: {e}")
                import traceback
                traceback.print_exc()
                self.timeline[frame_num]["gaze"] = None
            
            # Progress
            if (idx + 1) % 15 == 0 or idx == len(frames_to_process) - 1:
                progress = (idx + 1) / len(frames_to_process) * 100
                print(f"  Progress: {idx + 1}/{len(frames_to_process)} ({progress:.1f}%)")
        
        print(f"  ✓ Gaze estimation complete!")
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics from timeline."""
        # Collect data
        emotion_dist = {}
        total_blinks = 0
        pupil_sizes = []
        gaze_counts = {"left": 0, "center": 0, "right": 0}
        
        for entry in self.timeline:
            if entry.get("emotion"):
                label = entry["emotion"]["label"]
                emotion_dist[label] = emotion_dist.get(label, 0) + 1
            
            if entry.get("blink"):
                total_blinks = max(total_blinks, entry["blink"]["cumulative_blinks"])
            
            if entry.get("pupil") and entry["pupil"]["avg"] > 0:
                pupil_sizes.append(entry["pupil"]["avg"])
            
            if entry.get("gaze"):
                gaze_counts[entry["gaze"]["direction"]] += 1
        
        # Calculate statistics
        duration = len(self.frames) / self.fps
        
        # Emotion
        dominant_emotion = max(emotion_dist.items(), key=lambda x: x[1])[0] if emotion_dist else "unknown"
        emotion_changes = 0
        prev_emotion = None
        for entry in self.timeline:
            if entry.get("emotion"):
                if prev_emotion and entry["emotion"]["label"] != prev_emotion:
                    emotion_changes += 1
                prev_emotion = entry["emotion"]["label"]
        
        # Blink rate
        blink_rate = (total_blinks / duration * 60.0) if duration > 0 else 0.0
        
        # Pupil stats
        avg_pupil = np.mean(pupil_sizes) if pupil_sizes else 0.0
        max_pupil = np.max(pupil_sizes) if pupil_sizes else 0.0
        min_pupil = np.min(pupil_sizes) if pupil_sizes else 0.0
        
        # Pupil dilation events
        pupil_dilation_events = 0
        if len(pupil_sizes) > 30:
            baseline = np.mean(pupil_sizes[:30])
            for size in pupil_sizes:
                if abs(size - baseline) / baseline > 0.15:
                    pupil_dilation_events += 1
        
        # Gaze
        total_gaze = sum(gaze_counts.values())
        gaze_pct = {
            "left": round((gaze_counts["left"] / total_gaze * 100) if total_gaze > 0 else 0, 2),
            "center": round((gaze_counts["center"] / total_gaze * 100) if total_gaze > 0 else 0, 2),
            "right": round((gaze_counts["right"] / total_gaze * 100) if total_gaze > 0 else 0, 2)
        }
        attention_score = gaze_pct["center"] / 100.0
        
        return {
            "duration_seconds": duration,
            "total_frames": len(self.frames),
            "emotion": {
                "distribution": emotion_dist,
                "dominant_emotion": dominant_emotion,
                "emotion_changes": emotion_changes
            },
            "blink": {
                "total_blinks": total_blinks,
                "blink_rate_per_minute": round(blink_rate, 2)
            },
            "pupil": {
                "avg_pupil_size": round(float(avg_pupil), 4),
                "max_pupil_size": round(float(max_pupil), 4),
                "min_pupil_size": round(float(min_pupil), 4),
                "pupil_dilation_events": pupil_dilation_events
            },
            "gaze": {
                "distribution_percentage": gaze_pct,
                "attention_score": round(attention_score, 3)
            }
        }
