"""
Test the Gaze Processor with live webcam feed.
"""

import sys
from pathlib import Path
import cv2
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processors.gaze_processor import GazeProcessor


def capture_webcam_frames(duration_seconds=5, fps=30):
    """
    Capture frames from webcam for testing.
    
    Args:
        duration_seconds: How long to capture
        fps: Target frames per second
        
    Returns:
        List of captured frames
    """
    print(f"\nCapturing webcam frames for {duration_seconds} seconds...")
    print("Instructions:")
    print("- Look LEFT for a few seconds")
    print("- Look CENTER (straight ahead) for a few seconds")
    print("- Look RIGHT for a few seconds")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return [], 0
    
    # Set capture properties
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps == 0:
        actual_fps = fps
    
    print(f"Webcam opened at {actual_fps} FPS")
    print("Recording... Look LEFT, CENTER, and RIGHT")
    
    frames = []
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        frames.append(frame.copy())
        frame_count += 1
        
        # Show live preview with instructions
        elapsed = time.time() - start_time
        instruction = ""
        if elapsed < duration_seconds / 3:
            instruction = "Look LEFT"
            color = (0, 0, 255)
        elif elapsed < 2 * duration_seconds / 3:
            instruction = "Look CENTER"
            color = (0, 255, 0)
        else:
            instruction = "Look RIGHT"
            color = (255, 0, 0)
        
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            instruction,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3
        )
        cv2.putText(
            display_frame,
            f"Time: {int(elapsed)}s / {duration_seconds}s",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow('Recording - Follow Instructions', display_frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Captured {len(frames)} frames in {time.time() - start_time:.2f}s")
    
    return frames, actual_fps


def test_live_webcam():
    """Test gaze processor with live webcam capture."""
    print("\n" + "="*60)
    print("GAZE PROCESSOR TEST - LIVE WEBCAM")
    print("="*60)
    
    # Capture frames from webcam
    frames, fps = capture_webcam_frames(duration_seconds=9, fps=30)
    
    if not frames:
        print("\nFailed to capture frames")
        return
    
    # Initialize processor
    print("\nInitializing gaze processor...")
    processor = GazeProcessor()
    
    # Process frames
    print(f"\nProcessing {len(frames)} frames...")
    result = processor.compute_gaze_summary(frames, fps)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nTIMELINE:")
    print(f"Total entries: {len(result['timeline'])}")
    
    # Show first few, middle, and last few entries
    print("\nFirst 3 entries:")
    for i, entry in enumerate(result['timeline'][:3]):
        print(f"\n  Entry {i}:")
        print(f"    Time: {entry['t']:.2f}s")
        print(f"    Frame: {entry['frame']}")
        print(f"    Face detected: {entry['face_detected']}")
        print(f"    Eyes detected: {entry['eyes_detected']}")
        print(f"    Gaze direction: {entry['gaze_direction']}")
        print(f"    Left %: {entry['avg_left_perc']:.1f}, Right %: {entry['avg_right_perc']:.1f}")
    
    print("\nLast 3 entries:")
    for i, entry in enumerate(result['timeline'][-3:]):
        idx = len(result['timeline']) - 3 + i
        print(f"\n  Entry {idx}:")
        print(f"    Time: {entry['t']:.2f}s")
        print(f"    Frame: {entry['frame']}")
        print(f"    Gaze direction: {entry['gaze_direction']}")
        print(f"    Left %: {entry['avg_left_perc']:.1f}, Right %: {entry['avg_right_perc']:.1f}")
    
    print("\n" + "-"*60)
    print("SUMMARY:")
    print("-"*60)
    summary = result['summary']
    
    print(f"\nTotal frames: {summary['total_frames']}")
    print(f"Duration: {summary['duration_seconds']}s")
    print(f"Successful detections: {summary['successful_detections']}")
    print(f"Detection rate: {summary['detection_rate']:.2f}%")
    
    print(f"\nDominant direction: {summary['dominant_direction']}")
    print(f"Attention score (center focus): {summary['attention_score']:.1%}")
    
    print(f"\nGaze distribution (counts):")
    print(f"  Left: {summary['distribution']['left']}")
    print(f"  Center: {summary['distribution']['center']}")
    print(f"  Right: {summary['distribution']['right']}")
    
    print(f"\nGaze distribution (percentages):")
    print(f"  Left: {summary['distribution_percentage']['left']:.1f}%")
    print(f"  Center: {summary['distribution_percentage']['center']:.1f}%")
    print(f"  Right: {summary['distribution_percentage']['right']:.1f}%")
    
    print("\n" + "="*60)
    print("✓ TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        test_live_webcam()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
