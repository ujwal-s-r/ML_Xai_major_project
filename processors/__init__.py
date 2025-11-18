"""
Processors package for video analysis.
Contains high-level processors that wrap context models.
"""

from .blink_counter import BlinkCounterProcessor, compute_blink_summary
from .gaze_processor import GazeProcessor
from .pupil_processor import PupilProcessor
from .emotion_processor import EmotionProcessor
from .video_pipeline import VideoAnalysisPipeline, process_video

__all__ = [
    'BlinkCounterProcessor',
    'compute_blink_summary',
    'GazeProcessor',
    'PupilProcessor',
    'EmotionProcessor',
    'VideoAnalysisPipeline',
    'process_video',
]
