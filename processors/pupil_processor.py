"""
Pupil Processor
Wraps IrisTracker to provide continuous timeline data and summary metrics.
"""

import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from context.iris_tracker import IrisTracker


class PupilProcessor:
    """
    High-level processor for pupil/iris tracking.
    Generates continuous timeline data and aggregate summary.
    """
    
    def __init__(self):
        """Initialize the pupil processor."""
        self.iris_tracker = IrisTracker()
    
    def compute_pupil_summary(self, frames: List[np.ndarray], fps: float) -> Dict:
        """
        Process frames through iris tracker and generate timeline + summary.
        
        Args:
            frames: List of video frames (BGR format)
            fps: Frames per second
            
        Returns:
            Dictionary with:
            - timeline: List of per-frame pupil data
            - summary: Aggregate pupil metrics
        """
        print(f"[PupilProcessor] Processing {len(frames)} frames for pupil tracking...")
        
        # Reset tracker
        self.iris_tracker.reset()
        
        timeline = []
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Calculate timestamp
            timestamp = frame_idx / fps
            
            # Detect iris
            iris_data = self.iris_tracker.detect_iris(frame)
            
            # Build timeline entry
            timeline_entry = {
                't': round(timestamp, 3),
                'frame': frame_idx,
                'success': iris_data['success'],
                'left_iris_center': iris_data['left_iris_center'].tolist() if iris_data['left_iris_center'] is not None else None,
                'right_iris_center': iris_data['right_iris_center'].tolist() if iris_data['right_iris_center'] is not None else None,
                'left_pupil_size': float(iris_data['left_pupil_size']) if iris_data['left_pupil_size'] is not None else None,
                'right_pupil_size': float(iris_data['right_pupil_size']) if iris_data['right_pupil_size'] is not None else None,
                'avg_pupil_size': float((iris_data['left_pupil_size'] + iris_data['right_pupil_size']) / 2) if iris_data['left_pupil_size'] and iris_data['right_pupil_size'] else None,
                'pupil_dilation_ratio': float(iris_data['pupil_dilation_ratio']) if iris_data['pupil_dilation_ratio'] is not None else None
            }
            
            timeline.append(timeline_entry)
        
        # Get final metrics for summary
        metrics = self.iris_tracker.get_metrics()
        
        # Calculate summary statistics
        total_frames = len(frames)
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Count successful detections
        successful_detections = sum(1 for entry in timeline if entry['success'])
        detection_rate = (successful_detections / total_frames * 100) if total_frames > 0 else 0
        
        # Count pupil dilation events (significant changes from baseline)
        dilation_threshold = 0.1  # 10% change from baseline
        pupil_dilation_events = 0
        pupil_constriction_events = 0
        
        for entry in timeline:
            if entry['pupil_dilation_ratio'] is not None:
                if entry['pupil_dilation_ratio'] > (1.0 + dilation_threshold):
                    pupil_dilation_events += 1
                elif entry['pupil_dilation_ratio'] < (1.0 - dilation_threshold):
                    pupil_constriction_events += 1
        
        # Build summary
        summary = {
            'total_frames': total_frames,
            'duration_seconds': round(duration_seconds, 2),
            'successful_detections': successful_detections,
            'detection_rate': round(detection_rate, 2),
            'avg_pupil_size': round(float(metrics['avg_pupil_size']), 4) if metrics['avg_pupil_size'] > 0 else 0.0,
            'min_pupil_size': round(float(metrics['min_pupil_size']), 4) if metrics['min_pupil_size'] != float('inf') else 0.0,
            'max_pupil_size': round(float(metrics['max_pupil_size']), 4) if metrics['max_pupil_size'] > 0 else 0.0,
            'pupil_dilation_delta': round(float(metrics['pupil_dilation_delta']), 4),
            'pupil_dilation_events': pupil_dilation_events,
            'pupil_constriction_events': pupil_constriction_events,
            'baseline_recorded': metrics['baseline_recorded'],
            'baseline_pupil_size': round(float(metrics['baseline_pupil_size']), 4) if metrics['baseline_pupil_size'] is not None else None,
            'pupil_variability': {
                'mean': round(float(metrics['pupil_variability']['mean']), 4) if metrics['pupil_variability']['mean'] is not None else None,
                'std': round(float(metrics['pupil_variability']['std']), 4) if metrics['pupil_variability']['std'] is not None else None,
                'min': round(float(metrics['pupil_variability']['min']), 4) if metrics['pupil_variability']['min'] is not None else None,
                'max': round(float(metrics['pupil_variability']['max']), 4) if metrics['pupil_variability']['max'] is not None else None,
                'range': round(float(metrics['pupil_variability']['range']), 4) if metrics['pupil_variability']['range'] is not None else None
            }
        }
        
        print(f"[PupilProcessor] ✓ Processing complete")
        print(f"[PupilProcessor]   Average pupil size: {summary['avg_pupil_size']:.4f}px")
        print(f"[PupilProcessor]   Dilation delta: {summary['pupil_dilation_delta']:.4f}")
        print(f"[PupilProcessor]   Dilation events: {summary['pupil_dilation_events']}")
        print(f"[PupilProcessor]   Detection rate: {detection_rate:.1f}%")
        
        return {
            'timeline': timeline,
            'summary': summary
        }
    
    def reset(self):
        """Reset the processor to clean state."""
        self.iris_tracker.reset()
