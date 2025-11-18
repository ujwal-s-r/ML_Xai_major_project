"""
Test the complete video processing pipeline
"""

import sys
from pathlib import Path
import cv2
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processors.video_pipeline import VideoAnalysisPipeline


def capture_test_video(duration_seconds=5, output_path="test_video.mp4"):
    """Capture a short test video from webcam."""
    print(f"Opening webcam to capture {duration_seconds}s test video...")
    print("BLINK SEVERAL TIMES DELIBERATELY during recording!")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return None
    
    # Video writer setup
    fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Use H264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Recording... (Blink a few times)")
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1
            
            # Show live preview
            cv2.putText(frame, f"Recording: {int(time.time() - start_time)}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Recording...', frame)
            cv2.waitKey(1)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Captured {frame_count} frames to {output_path}")
    return output_path


def test_pipeline():
    """Test the complete video processing pipeline."""
    print("\n" + "="*60)
    print("VIDEO PROCESSING PIPELINE TEST")
    print("="*60 + "\n")
    
    # Capture test video
    video_path = capture_test_video(duration_seconds=5)
    
    if not video_path:
        print("FAILED: Could not capture test video")
        return
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = VideoAnalysisPipeline()
    
    # Process video
    print(f"\nProcessing video: {video_path}")
    results = pipeline.process_video_file(video_path)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nTimeline: {len(results['timeline'])} entries")
    
    if results['timeline']:
        print("\nFirst 3 timeline entries:")
        for i, entry in enumerate(results['timeline'][:3]):
            print(f"\n  Entry {i}:")
            print(f"    Time: {entry['t']:.2f}s")
            print(f"    Frame: {entry['frame']}")
            print(f"    Blink data:")
            print(f"      - EAR: {entry['blink']['avg_ear']:.3f}")
            print(f"      - Is blinking: {entry['blink']['is_blinking']}")
            print(f"      - Total blinks: {entry['blink']['blink_count']}")
    
    print(f"\nSummary:")
    print(f"  Blink Analysis:")
    for key, value in results['summary']['blink'].items():
        print(f"    - {key}: {value}")
    
    # Clean up
    import os
    try:
        os.remove(video_path)
        print(f"\n✓ Cleaned up test video")
    except:
        pass
    
    print("\n" + "="*60)
    print("✓ PIPELINE TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
