"""
Blink Counter Processor
Provides a clean API for blink detection using BlinkDetector from context.
"""

import numpy as np
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from context.blink_detector import BlinkDetector


class BlinkCounterProcessor:
    """
    High-level processor for blink detection across video frames.
    Wraps BlinkDetector and provides timeline + summary output.
    """
    
    def __init__(self):
        """Initialize the blink counter processor."""
        self.detector = BlinkDetector()
    
    def compute_blink_summary(
        self, 
        frames: List[np.ndarray], 
        fps: float = 30.0
    ) -> Dict:
        """
        Process a sequence of frames and return blink analysis.
        
        Args:
            frames: List of video frames (numpy arrays in BGR format)
            fps: Frames per second of the video
            
        Returns:
            Dictionary containing:
            - timeline: List of per-frame blink data
            - summary: Aggregate blink metrics
        """
        if not frames:
            return {
                'timeline': [],
                'summary': {
                    'total_blinks': 0,
                    'blink_rate_per_minute': 0.0,
                    'avg_ear': 0.0,
                    'total_frames': 0,
                    'duration_seconds': 0.0
                }
            }
        
        # Reset detector state
        self.detector.reset()
        self.detector.fps = fps
        
        timeline = []
        ear_values = []
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            timestamp = frame_idx / fps
            
            # Detect blink in this frame
            blink_data = self.detector.detect_blink(frame)
            
            # Build timeline entry
            timeline_entry = {
                't': round(timestamp, 3),
                'frame': frame_idx,
                'left_ear': round(blink_data['left_ear'], 4),
                'right_ear': round(blink_data['right_ear'], 4),
                'avg_ear': round(blink_data['avg_ear'], 4),
                'is_blinking': blink_data['is_blinking'],
                'blink_count': blink_data['blink_count'],
                'success': blink_data['success']
            }
            
            timeline.append(timeline_entry)
            
            if blink_data['success']:
                ear_values.append(blink_data['avg_ear'])
        
        # Compute summary statistics
        total_blinks = self.detector.blink_counter
        duration_seconds = len(frames) / fps
        
        # Calculate blink rate
        if duration_seconds > 0:
            blink_rate_per_minute = (total_blinks / duration_seconds) * 60.0
        else:
            blink_rate_per_minute = 0.0
        
        # Average EAR across all successful detections
        avg_ear = np.mean(ear_values) if ear_values else 0.0
        
        summary = {
            'total_blinks': total_blinks,
            'blink_rate_per_minute': round(blink_rate_per_minute, 2),
            'avg_ear': round(float(avg_ear), 4),
            'total_frames': len(frames),
            'duration_seconds': round(duration_seconds, 2),
            'successful_detections': len(ear_values),
            'detection_rate': round(len(ear_values) / len(frames) * 100, 2) if frames else 0.0
        }
        
        return {
            'timeline': timeline,
            'summary': summary
        }
    
    def reset(self):
        """Reset the detector state."""
        self.detector.reset()


def compute_blink_summary(frames: List[np.ndarray], fps: float = 30.0) -> Dict:
    """
    Convenience function for blink analysis.
    
    Args:
        frames: List of video frames (numpy arrays in BGR format)
        fps: Frames per second
        
    Returns:
        Dictionary with timeline and summary
    """
    processor = BlinkCounterProcessor()
    return processor.compute_blink_summary(frames, fps)
