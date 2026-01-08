"""
Gaze Processor
Wraps GazeEstimator to provide continuous timeline data and summary metrics.
"""

import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from context.gaze_estimator import GazeEstimator


class GazeProcessor:
    """
    High-level processor for gaze estimation.
    Generates continuous timeline data and aggregate summary.
    """
    
    def __init__(self):
        """Initialize the gaze processor."""
        self.gaze_estimator = GazeEstimator()
    
    def compute_gaze_summary(self, frames: List[np.ndarray], fps: float) -> Dict:
        """
        Process frames through gaze estimator and generate timeline + summary.
        
        Args:
            frames: List of video frames (BGR format)
            fps: Frames per second
            
        Returns:
            Dictionary with:
            - timeline: List of per-frame gaze data
            - summary: Aggregate gaze metrics
        """
        print(f"[GazeProcessor] Processing {len(frames)} frames for gaze estimation...")
        
        # Reset counters
        self.gaze_estimator.reset()
        
        timeline = []
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Analyze frame
            _, metrics = self.gaze_estimator.analyze_frame(frame)
            
            # Build timeline entry
            timeline_entry = {
                't': round(timestamp, 3),
                'frame': frame_idx,
                'face_detected': metrics['face_detected'],
                'eyes_detected': metrics['eyes_detected'],
                'gaze_direction': metrics['gaze_direction'],
                'avg_left_perc': metrics['avg_left_perc'],
                'avg_right_perc': metrics['avg_right_perc'],
                'left_pupil_coords': metrics['left_pupil_coords'],
                'right_pupil_coords': metrics['right_pupil_coords']
            }
            
            timeline.append(timeline_entry)
        
        # Get final metrics for summary
        final_metrics = self.gaze_estimator.get_last_metrics()
        
        # Calculate summary statistics
        total_frames = len(frames)
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Calculate percentages for each gaze direction
        left_percentage = (final_metrics['looking_left_count'] / total_frames * 100) if total_frames > 0 else 0
        center_percentage = (final_metrics['looking_center_count'] / total_frames * 100) if total_frames > 0 else 0
        right_percentage = (final_metrics['looking_right_count'] / total_frames * 100) if total_frames > 0 else 0
        
        # Determine dominant gaze direction
        max_count = max(
            final_metrics['looking_left_count'],
            final_metrics['looking_center_count'],
            final_metrics['looking_right_count']
        )
        
        if max_count == final_metrics['looking_left_count']:
            dominant_direction = "Looking Left"
        elif max_count == final_metrics['looking_center_count']:
            dominant_direction = "Looking Center"
        else:
            dominant_direction = "Looking Right"
        
        # Calculate attention score (percentage of time looking at center)
        attention_score = center_percentage / 100.0
        
        # Count successful detections
        successful_detections = sum(1 for entry in timeline if entry['face_detected'] and entry['eyes_detected'])
        detection_rate = (successful_detections / total_frames * 100) if total_frames > 0 else 0
        
        # Build summary
        summary = {
            'total_frames': total_frames,
            'duration_seconds': round(duration_seconds, 2),
            'successful_detections': successful_detections,
            'detection_rate': round(detection_rate, 2),
            'dominant_direction': dominant_direction,
            'attention_score': round(attention_score, 3),
            'distribution': {
                'left': final_metrics['looking_left_count'],
                'center': final_metrics['looking_center_count'],
                'right': final_metrics['looking_right_count']
            },
            'distribution_percentage': {
                'left': round(left_percentage, 2),
                'center': round(center_percentage, 2),
                'right': round(right_percentage, 2)
            }
        }
        
        print(f"[GazeProcessor] ✓ Processing complete")
        print(f"[GazeProcessor]   Dominant direction: {dominant_direction}")
        print(f"[GazeProcessor]   Attention score: {attention_score:.1%}")
        print(f"[GazeProcessor]   Detection rate: {detection_rate:.1f}%")
        
        return {
            'timeline': timeline,
            'summary': summary
        }
    
    def reset(self):
        """Reset the processor to clean state."""
        self.gaze_estimator.reset()
