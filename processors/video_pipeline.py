"""
Video Processing Pipeline
Orchestrates all video analysis models and generates timeline + summary.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from processors.blink_counter import BlinkCounterProcessor
from processors.gaze_processor import GazeProcessor
from processors.pupil_processor import PupilProcessor
from processors.emotion_processor import EmotionProcessor


class VideoAnalysisPipeline:
    """
    Main pipeline for processing uploaded videos through all ML models.
    Generates continuous timeline data and aggregate summaries.
    """
    
    def __init__(self):
        """Initialize all model processors."""
        print("[Pipeline] Initializing video analysis pipeline...")
        
        # Initialize blink counter
        self.blink_processor = BlinkCounterProcessor()
        print("[Pipeline] ✓ Blink counter initialized")
        
        # Initialize gaze processor
        self.gaze_processor = GazeProcessor()
        print("[Pipeline] ✓ Gaze processor initialized")
        
        # Initialize pupil processor
        self.pupil_processor = PupilProcessor()
        print("[Pipeline] ✓ Pupil processor initialized")
        
        # Initialize emotion processor
        self.emotion_processor = EmotionProcessor()
        print("[Pipeline] ✓ Emotion processor initialized")
        
        print("[Pipeline] Pipeline ready")
    
    def process_video_file(self, video_path: str) -> Dict:
        """
        Process a video file through all models.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with timeline and summary data
        """
        print(f"[Pipeline] Processing video: {video_path}")
        
        # Extract frames from video
        frames, fps = self._extract_frames(video_path)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        print(f"[Pipeline] Extracted {len(frames)} frames at {fps} FPS")
        
        # Process through all models
        return self.process_frames(frames, fps)
    
    def process_frames(self, frames: List[np.ndarray], fps: float) -> Dict:
        """
        Process a list of frames through all models.
        
        Args:
            frames: List of video frames (BGR format)
            fps: Frames per second
            
        Returns:
            Dictionary with:
            - timeline: List of per-frame data from all models
            - summary: Aggregate metrics from all models
        """
        print(f"[Pipeline] Processing {len(frames)} frames through models...")
        
        # Process through blink counter
        print("[Pipeline] Running blink detection...")
        blink_result = self.blink_processor.compute_blink_summary(frames, fps)
        
        # Process through gaze estimator
        print("[Pipeline] Running gaze estimation...")
        gaze_result = self.gaze_processor.compute_gaze_summary(frames, fps)
        
        # Process through pupil tracker
        # Note: Pupil processor uses first 30 frames as baseline (automatic calibration)
        print("[Pipeline] Running pupil tracking...")
        pupil_result = self.pupil_processor.compute_pupil_summary(frames, fps)
        
        # Process through emotion analyzer
        print("[Pipeline] Running emotion analysis with XAI...")
        emotion_result = self.emotion_processor.compute_emotion_summary(frames, fps)
        
        # Merge timelines from all models
        timeline = self._merge_timelines(
            blink_timeline=blink_result['timeline'],
            gaze_timeline=gaze_result['timeline'],
            pupil_timeline=pupil_result['timeline'],
            emotion_timeline=emotion_result['timeline'],
        )
        
        # Build combined summary
        summary = {
            'blink': blink_result['summary'],
            'gaze': gaze_result['summary'],
            'pupil': pupil_result['summary'],
            'emotion': emotion_result['summary'],
        }
        
        print(f"[Pipeline] ✓ Processing complete")
        print(f"[Pipeline]   Timeline entries: {len(timeline)}")
        print(f"[Pipeline]   Total blinks: {summary['blink']['total_blinks']}")
        print(f"[Pipeline]   Dominant gaze: {summary['gaze']['dominant_direction']}")
        try:
            print(f"[Pipeline]   Avg pupil size: {summary['pupil']['avg_pupil_size']:.4f}px")
        except Exception:
            print("[Pipeline]   Avg pupil size: N/A")
        try:
            print(f"[Pipeline]   Dominant emotion: {summary.get('emotion', {}).get('dominant_emotion')}")
        except Exception:
            print("[Pipeline]   Dominant emotion: N/A")
        
        return {
            'timeline': timeline,
            'summary': summary
        }
    
    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """
        Extract all frames from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames list, fps)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0  # Default fallback
        
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        return frames, fps
    
    def _merge_timelines(
        self,
        blink_timeline: List[Dict],
        gaze_timeline: List[Dict] = None,
        pupil_timeline: List[Dict] = None,
        emotion_timeline: List[Dict] = None,
    ) -> List[Dict]:
        """
        Merge timelines from all models into a unified timeline.
        
        Each timeline entry will have:
        - t: timestamp in seconds
        - frame: frame index
        - blink: {...}  # blink detection data
        - gaze: {...}  # gaze estimation data
        - pupil: {...}  # pupil tracking data
        - emotion: {...}  # emotion analysis data (coming soon)
        
        Args:
            blink_timeline: Timeline from blink detector
            gaze_timeline: Timeline from gaze estimator
            pupil_timeline: Timeline from pupil tracker
            
        Returns:
            Merged timeline list
        """
        if not blink_timeline:
            return []
        
        merged = []
        
        for i, blink_entry in enumerate(blink_timeline):
            entry = {
                't': blink_entry['t'],
                'frame': blink_entry['frame'],
                'blink': {
                    'left_ear': blink_entry.get('left_ear'),
                    'right_ear': blink_entry.get('right_ear'),
                    'avg_ear': blink_entry.get('avg_ear'),
                    'is_blinking': blink_entry.get('is_blinking', False),
                    'blink_count': blink_entry.get('blink_count', 0),
                    'success': blink_entry.get('success', False)
                }
            }
            
            # Add gaze data
            if gaze_timeline and i < len(gaze_timeline):
                gaze_entry = gaze_timeline[i]
                entry['gaze'] = {
                    'face_detected': gaze_entry.get('face_detected', False),
                    'eyes_detected': gaze_entry.get('eyes_detected', False),
                    'gaze_direction': gaze_entry.get('gaze_direction'),
                    'avg_left_perc': gaze_entry.get('avg_left_perc'),
                    'avg_right_perc': gaze_entry.get('avg_right_perc'),
                    'left_pupil_coords': gaze_entry.get('left_pupil_coords'),
                    'right_pupil_coords': gaze_entry.get('right_pupil_coords')
                }
            
            # Add pupil data
            if pupil_timeline and i < len(pupil_timeline):
                pupil_entry = pupil_timeline[i]
                entry['pupil'] = {
                    'success': pupil_entry.get('success', False),
                    'left_iris_center': pupil_entry.get('left_iris_center'),
                    'right_iris_center': pupil_entry.get('right_iris_center'),
                    'left_pupil_size': pupil_entry.get('left_pupil_size'),
                    'right_pupil_size': pupil_entry.get('right_pupil_size'),
                    'avg_pupil_size': pupil_entry.get('avg_pupil_size'),
                    'pupil_dilation_ratio': pupil_entry.get('pupil_dilation_ratio')
                }
            
            # Add emotion data (align with EmotionProcessor timeline schema)
            if emotion_timeline and i < len(emotion_timeline):
                emotion_entry = emotion_timeline[i]
                entry['emotion'] = {
                    'success': emotion_entry.get('success', False),
                    'dominant_emotion': emotion_entry.get('dominant_emotion'),
                    'dominant_emotion_code': emotion_entry.get('dominant_emotion_code'),
                    'emotions': emotion_entry.get('emotions', {}),
                    'has_xai': emotion_entry.get('has_xai', False)
                }
                # Add XAI data if present (keys are optional)
                if emotion_entry.get('has_xai'):
                    xai_data = {}
                    if 'attention_map' in emotion_entry:
                        xai_data['attention_map'] = emotion_entry['attention_map']
                        xai_data['attention_grid_size'] = emotion_entry.get('attention_grid_size')
                    if 'gradcam_heatmap' in emotion_entry:
                        xai_data['gradcam_heatmap'] = emotion_entry['gradcam_heatmap']
                        xai_data['gradcam_target'] = emotion_entry.get('gradcam_target')
                    if xai_data:
                        entry['emotion']['xai'] = xai_data
            
            merged.append(entry)
        
        return merged
    
    def reset(self):
        """Reset all processors to clean state."""
        self.blink_processor.reset()
        self.gaze_processor.reset()
        self.pupil_processor.reset()
        self.emotion_processor.reset()


# Convenience function for one-shot processing
def process_video(video_path: str) -> Dict:
    """
    Process a video file through the complete pipeline.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with timeline and summary
    """
    pipeline = VideoAnalysisPipeline()
    return pipeline.process_video_file(video_path)
