"""
Unified Video Analyzer integrating all 4 analysis models:
- Emotion Analysis (Transformers ViT)
- Blink Detection (MediaPipe Face Mesh)
- Iris/Pupil Tracking (MediaPipe Iris)
- Gaze Estimation (L2CS-Net)

Processes frames in real-time with optimized sampling rates.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import pathlib
import torch

from .emotion_analyzer import EmotionAnalyzer
from .blink_detector import BlinkDetector
from .iris_tracker import IrisTracker

# L2CS imports
from l2cs import Pipeline, select_device


class VideoAnalyzer:
    """
    Unified analyzer that processes video frames through all 4 models
    with optimized frame sampling and time-series data collection.
    """
    
    def __init__(self, 
                 emotion_sample_rate: int = 10,  # Process every 10th frame (3 FPS at 30fps)
                 blink_sample_rate: int = 2,     # Process every 2nd frame (15 FPS at 30fps)
                 iris_sample_rate: int = 2,      # Process every 2nd frame (15 FPS at 30fps)
                 gaze_sample_rate: int = 3):     # Process every 3rd frame (10 FPS at 30fps)
        """
        Initialize all analyzers with specified sampling rates.
        
        Args:
            emotion_sample_rate: Process emotion every N frames
            blink_sample_rate: Process blink every N frames
            iris_sample_rate: Process iris every N frames
            gaze_sample_rate: Process gaze every N frames
        """
        print("Initializing VideoAnalyzer...")
        
        # Sampling configuration
        self.emotion_sample_rate = emotion_sample_rate
        self.blink_sample_rate = blink_sample_rate
        self.iris_sample_rate = iris_sample_rate
        self.gaze_sample_rate = gaze_sample_rate
        
        # Initialize all analyzers
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
        
        # Time-series data storage
        self.timeline: List[Dict] = []
        self.frame_count = 0
        self.start_time = None
        self.fps = 30.0  # Default FPS
        
        # Summary statistics
        self.emotion_distribution = {}
        self.total_blinks = 0
        self.pupil_sizes = []
        self.gaze_directions = {"left": 0, "center": 0, "right": 0}
        
        print("✓ VideoAnalyzer initialized successfully!")
    
    def start_analysis(self):
        """Start the analysis session (reset counters and timeline)."""
        self.timeline = []
        self.frame_count = 0
        self.start_time = time.time()
        self.emotion_distribution = {}
        self.total_blinks = 0
        self.pupil_sizes = []
        self.gaze_directions = {"left": 0, "center": 0, "right": 0}
        
        # Reset individual analyzers
        self.blink_detector.reset()
        self.iris_tracker.reset()
        
        print("Analysis session started!")
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """
        Process a single frame through appropriate models based on sampling rates.
        
        Args:
            frame: BGR image from webcam
            frame_number: Sequential frame number
            
        Returns:
            Dictionary with analysis results and processing flags
        """
        self.frame_count = frame_number
        timestamp = frame_number / self.fps
        
        # Initialize result structure
        result = {
            "timestamp": timestamp,
            "frame_number": frame_number,
            "processed": {
                "emotion": False,
                "blink": False,
                "iris": False,
                "gaze": False
            }
        }
        
        # Process Emotion (least frequent - every 10 frames)
        if frame_number % self.emotion_sample_rate == 0:
            try:
                emotion_result = self.emotion_analyzer.analyze_emotion(frame)
                result["emotion"] = {
                    "label": emotion_result.get("dominant_emotion", "unknown"),
                    "confidence": emotion_result.get("confidence", 0.0),
                    "scores": emotion_result.get("all_emotions", {})
                }
                result["processed"]["emotion"] = True
                
                # Update distribution
                label = result["emotion"]["label"]
                self.emotion_distribution[label] = self.emotion_distribution.get(label, 0) + 1
                
            except Exception as e:
                print(f"Emotion analysis error at frame {frame_number}: {e}")
                result["emotion"] = None
        
        # Process Blink (every 2 frames)
        if frame_number % self.blink_sample_rate == 0:
            try:
                blink_result = self.blink_detector.detect_blink(frame)
                if blink_result.get("success"):
                    result["blink"] = {
                        "ear_left": blink_result.get("left_ear", 0.0),
                        "ear_right": blink_result.get("right_ear", 0.0),
                        "avg_ear": blink_result.get("avg_ear", 0.0),
                        "is_blinking": blink_result.get("is_blinking", False),
                        "cumulative_blinks": blink_result.get("blink_count", 0)
                    }
                    result["processed"]["blink"] = True
                    self.total_blinks = result["blink"]["cumulative_blinks"]
                else:
                    result["blink"] = None
                    
            except Exception as e:
                print(f"Blink detection error at frame {frame_number}: {e}")
                result["blink"] = None
        
        # Process Iris/Pupil (every 2 frames, can share MediaPipe with blink)
        if frame_number % self.iris_sample_rate == 0:
            try:
                iris_result = self.iris_tracker.detect_iris(frame)
                if iris_result.get("success"):
                    left_size = iris_result.get("left_pupil_size", 0.0)
                    right_size = iris_result.get("right_pupil_size", 0.0)
                    avg_size = (left_size + right_size) / 2.0 if left_size and right_size else 0.0
                    
                    result["pupil"] = {
                        "left": left_size,
                        "right": right_size,
                        "avg": avg_size,
                        "dilation_ratio": iris_result.get("pupil_dilation_ratio", 0.0)
                    }
                    result["processed"]["iris"] = True
                    
                    if avg_size > 0:
                        self.pupil_sizes.append(avg_size)
                else:
                    result["pupil"] = None
                    
            except Exception as e:
                print(f"Iris tracking error at frame {frame_number}: {e}")
                result["pupil"] = None
        
        # Process Gaze (every 3 frames)
        if frame_number % self.gaze_sample_rate == 0:
            try:
                gaze_result = self.gaze_pipeline.step(frame)
                
                if gaze_result and gaze_result.get('pitch') and gaze_result.get('yaw'):
                    pitch = gaze_result['pitch'][0] if isinstance(gaze_result['pitch'], list) else gaze_result['pitch']
                    yaw = gaze_result['yaw'][0] if isinstance(gaze_result['yaw'], list) else gaze_result['yaw']
                    
                    # Determine direction
                    if yaw < -15:
                        direction = "left"
                    elif yaw > 15:
                        direction = "right"
                    else:
                        direction = "center"
                    
                    result["gaze"] = {
                        "pitch": float(pitch),
                        "yaw": float(yaw),
                        "direction": direction
                    }
                    result["processed"]["gaze"] = True
                    
                    # Update direction distribution
                    self.gaze_directions[direction] += 1
                else:
                    result["gaze"] = None
                    
            except Exception as e:
                print(f"Gaze estimation error at frame {frame_number}: {e}")
                result["gaze"] = None
        
        # Add to timeline
        self.timeline.append(result)
        
        return result
    
    def get_summary(self) -> Dict:
        """
        Generate summary statistics from all collected data.
        
        Returns:
            Dictionary with comprehensive summary metrics
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        duration = self.frame_count / self.fps
        
        # Emotion summary
        total_emotion_frames = sum(self.emotion_distribution.values())
        dominant_emotion = max(self.emotion_distribution.items(), key=lambda x: x[1])[0] if self.emotion_distribution else "unknown"
        
        # Count emotion changes
        emotion_changes = 0
        prev_emotion = None
        for entry in self.timeline:
            if entry.get("emotion"):
                curr_emotion = entry["emotion"]["label"]
                if prev_emotion and curr_emotion != prev_emotion:
                    emotion_changes += 1
                prev_emotion = curr_emotion
        
        # Blink summary
        blink_rate = (self.total_blinks / duration * 60.0) if duration > 0 else 0.0
        
        # Pupil summary
        avg_pupil = np.mean(self.pupil_sizes) if self.pupil_sizes else 0.0
        max_pupil = np.max(self.pupil_sizes) if self.pupil_sizes else 0.0
        min_pupil = np.min(self.pupil_sizes) if self.pupil_sizes else 0.0
        
        # Calculate pupil dilation events (significant changes)
        pupil_dilation_events = 0
        if len(self.pupil_sizes) > 1:
            baseline = np.mean(self.pupil_sizes[:30]) if len(self.pupil_sizes) >= 30 else np.mean(self.pupil_sizes)
            for size in self.pupil_sizes:
                if abs(size - baseline) / baseline > 0.15:  # 15% change
                    pupil_dilation_events += 1
        
        # Gaze summary
        total_gaze_samples = sum(self.gaze_directions.values())
        gaze_time_distribution = {}
        if total_gaze_samples > 0:
            gaze_time_distribution = {
                "left": (self.gaze_directions["left"] / total_gaze_samples) * duration,
                "center": (self.gaze_directions["center"] / total_gaze_samples) * duration,
                "right": (self.gaze_directions["right"] / total_gaze_samples) * duration
            }
        
        attention_score = gaze_time_distribution.get("center", 0) / duration if duration > 0 else 0.0
        
        return {
            "duration_seconds": duration,
            "total_frames": self.frame_count,
            "processing_time_seconds": elapsed_time,
            
            "emotion": {
                "distribution": dict(self.emotion_distribution),
                "dominant_emotion": dominant_emotion,
                "emotion_changes": emotion_changes,
                "total_samples": total_emotion_frames
            },
            
            "blink": {
                "total_blinks": self.total_blinks,
                "blink_rate_per_minute": round(blink_rate, 2),
                "avg_blink_rate": round(blink_rate, 2)
            },
            
            "pupil": {
                "avg_pupil_size": round(float(avg_pupil), 4),
                "max_pupil_size": round(float(max_pupil), 4),
                "min_pupil_size": round(float(min_pupil), 4),
                "pupil_dilation_events": pupil_dilation_events,
                "total_samples": len(self.pupil_sizes)
            },
            
            "gaze": {
                "distribution_seconds": gaze_time_distribution,
                "distribution_percentage": {
                    "left": round((self.gaze_directions["left"] / total_gaze_samples * 100) if total_gaze_samples > 0 else 0, 2),
                    "center": round((self.gaze_directions["center"] / total_gaze_samples * 100) if total_gaze_samples > 0 else 0, 2),
                    "right": round((self.gaze_directions["right"] / total_gaze_samples * 100) if total_gaze_samples > 0 else 0, 2)
                },
                "attention_score": round(attention_score, 3),
                "total_samples": total_gaze_samples
            }
        }
    
    def get_timeline_data(self) -> List[Dict]:
        """
        Get the complete time-series timeline data.
        
        Returns:
            List of timeline entries with all metrics
        """
        return self.timeline
    
    def export_results(self) -> Dict:
        """
        Export complete analysis results including timeline and summary.
        
        Returns:
            Dictionary with all analysis data ready for storage
        """
        return {
            "timeline": self.timeline,
            "summary": self.get_summary()
        }
