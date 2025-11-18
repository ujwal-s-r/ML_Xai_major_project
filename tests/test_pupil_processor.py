"""
Test the Pupil Processor with live webcam feed.
"""

import sys
from pathlib import Path
import cv2
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processors.pupil_processor import PupilProcessor


def capture_webcam_frames(duration_seconds=10, fps=30):
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
    print("- First 3 seconds: Look at the camera normally (baseline)")
    print("- Next 3 seconds: Look at something bright (dilate pupils)")
    print("- Last 4 seconds: Look back at camera normally")
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
    print("Recording...")
    
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
        
        if elapsed < 3:
            instruction = "BASELINE: Look at camera normally"
            color = (0, 255, 0)
        elif elapsed < 6:
            instruction = "Look at something BRIGHT"
            color = (0, 255, 255)
        else:
            instruction = "Look back at camera"
            color = (0, 255, 0)
        
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            instruction,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        cv2.putText(
            display_frame,
            f"Time: {int(elapsed)}s / {duration_seconds}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
    """Test pupil processor with live webcam capture."""
    print("\n" + "="*60)
    print("PUPIL PROCESSOR TEST - LIVE WEBCAM")
    print("="*60)
    
    # Capture frames from webcam
    frames, fps = capture_webcam_frames(duration_seconds=10, fps=30)
    
    if not frames:
        print("\nFailed to capture frames")
        return
    
    # Initialize processor
    print("\nInitializing pupil processor...")
    processor = PupilProcessor()
    
    # Process frames
    print(f"\nProcessing {len(frames)} frames...")
    result = processor.compute_pupil_summary(frames, fps)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nTIMELINE:")
    print(f"Total entries: {len(result['timeline'])}")
    
    # Show first few entries (baseline)
    print("\nFirst 5 entries (BASELINE):")
    for i, entry in enumerate(result['timeline'][:5]):
        print(f"\n  Entry {i}:")
        print(f"    Time: {entry['t']:.2f}s")
        print(f"    Frame: {entry['frame']}")
        print(f"    Success: {entry['success']}")
        if entry['avg_pupil_size']:
            print(f"    Avg pupil size: {entry['avg_pupil_size']:.4f}px")
            if entry['pupil_dilation_ratio']:
                print(f"    Dilation ratio: {entry['pupil_dilation_ratio']:.4f}")
    
    # Show middle entries (event)
    mid_point = len(result['timeline']) // 2
    print(f"\nMiddle 5 entries (around frame {mid_point}):")
    for i in range(mid_point - 2, mid_point + 3):
        if i < len(result['timeline']):
            entry = result['timeline'][i]
            print(f"\n  Entry {i}:")
            print(f"    Time: {entry['t']:.2f}s")
            if entry['avg_pupil_size']:
                print(f"    Avg pupil size: {entry['avg_pupil_size']:.4f}px")
                if entry['pupil_dilation_ratio']:
                    print(f"    Dilation ratio: {entry['pupil_dilation_ratio']:.4f}")
    
    # Show last few entries
    print("\nLast 5 entries:")
    for i, entry in enumerate(result['timeline'][-5:]):
        idx = len(result['timeline']) - 5 + i
        print(f"\n  Entry {idx}:")
        print(f"    Time: {entry['t']:.2f}s")
        if entry['avg_pupil_size']:
            print(f"    Avg pupil size: {entry['avg_pupil_size']:.4f}px")
            if entry['pupil_dilation_ratio']:
                print(f"    Dilation ratio: {entry['pupil_dilation_ratio']:.4f}")
    
    print("\n" + "-"*60)
    print("SUMMARY:")
    print("-"*60)
    summary = result['summary']
    
    print(f"\nTotal frames: {summary['total_frames']}")
    print(f"Duration: {summary['duration_seconds']}s")
    print(f"Successful detections: {summary['successful_detections']}")
    print(f"Detection rate: {summary['detection_rate']:.2f}%")
    
    print(f"\nBaseline recorded: {summary['baseline_recorded']}")
    if summary['baseline_pupil_size']:
        print(f"Baseline pupil size: {summary['baseline_pupil_size']:.4f}px")
    
    print(f"\nPupil size metrics:")
    print(f"  Average: {summary['avg_pupil_size']:.4f}px")
    print(f"  Minimum: {summary['min_pupil_size']:.4f}px")
    print(f"  Maximum: {summary['max_pupil_size']:.4f}px")
    
    print(f"\nPupil dilation analysis:")
    print(f"  Dilation delta: {summary['pupil_dilation_delta']:.4f}")
    print(f"  Dilation events: {summary['pupil_dilation_events']}")
    print(f"  Constriction events: {summary['pupil_constriction_events']}")
    
    print(f"\nPupil variability:")
    var = summary['pupil_variability']
    if var['mean']:
        print(f"  Mean: {var['mean']:.4f}px")
        print(f"  Std Dev: {var['std']:.4f}px")
        print(f"  Min: {var['min']:.4f}px")
        print(f"  Max: {var['max']:.4f}px")
        print(f"  Range: {var['range']:.4f}px")
    
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
