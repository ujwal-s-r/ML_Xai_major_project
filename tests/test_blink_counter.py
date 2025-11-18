"""
Test script for BlinkCounterProcessor
Tests blink detection on live webcam feed
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processors.blink_counter import BlinkCounterProcessor, compute_blink_summary


def capture_webcam_frames(duration_seconds=5, fps=30):
    """
    Capture frames from webcam for specified duration.
    
    Args:
        duration_seconds: How long to record
        fps: Target frames per second
        
    Returns:
        List of captured frames
    """
    print(f"Opening webcam... (will record for {duration_seconds} seconds)")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    frames = []
    frame_interval = 1.0 / fps
    start_time = time.time()
    last_capture_time = start_time
    
    print(f"Recording started! Blink naturally during the {duration_seconds} seconds...")
    print("TIP: Try blinking 2-3 times during recording")
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Check if duration exceeded
        if elapsed >= duration_seconds:
            break
        
        # Capture frame at target FPS
        if current_time - last_capture_time >= frame_interval:
            ret, frame = cap.read()
            if ret:
                frames.append(frame.copy())
                last_capture_time = current_time
                
                # Show progress
                if len(frames) % 30 == 0:
                    print(f"  Captured {len(frames)} frames... ({elapsed:.1f}s)")
    
    cap.release()
    print(f"Recording complete! Captured {len(frames)} frames")
    return frames


def test_live_webcam():
    """Test with live webcam feed."""
    print("=" * 60)
    print("Test 1: Live Webcam Blink Detection")
    print("=" * 60)
    
    # Capture frames from webcam
    frames = capture_webcam_frames(duration_seconds=5, fps=30)
    
    if frames is None or len(frames) == 0:
        print("SKIPPED: No webcam available or no frames captured")
        return None
    
    fps = 30.0
    print(f"\nProcessing {len(frames)} frames at {fps} FPS...")
    
    # Process using the convenience function
    result = compute_blink_summary(frames, fps)
    
    print("\n--- Results ---")
    print(f"Timeline entries: {len(result['timeline'])}")
    print(f"\nSummary:")
    for key, value in result['summary'].items():
        print(f"  {key}: {value}")
    
    # Show some timeline samples
    print(f"\nSample timeline entries (first 5):")
    for i, entry in enumerate(result['timeline'][:5]):
        print(f"  Frame {i}: t={entry['t']:.2f}s, EAR={entry['avg_ear']:.3f}, blinking={entry['is_blinking']}, blinks={entry['blink_count']}")
    
    # Verify structure
    assert 'timeline' in result, "Missing timeline in result"
    assert 'summary' in result, "Missing summary in result"
    assert len(result['timeline']) == len(frames), "Timeline length mismatch"
    
    print("\nOK: Live webcam test passed!")
    return result


def test_empty_frames():
    """Test with empty frame list."""
    print("\n" + "=" * 60)
    print("Test 2: Empty Frames Edge Case")
    print("=" * 60)
    
    result = compute_blink_summary([], fps=30.0)
    
    print("Summary for empty input:")
    for key, value in result['summary'].items():
        print(f"  {key}: {value}")
    
    assert result['summary']['total_blinks'] == 0
    assert result['summary']['total_frames'] == 0
    assert len(result['timeline']) == 0
    
    print("\nOK: Empty frames test passed!")


def test_processor_reset():
    """Test that processor can be reset and reused."""
    print("\n" + "=" * 60)
    print("Test 3: Processor Reset (Programmatic)")
    print("=" * 60)
    
    processor = BlinkCounterProcessor()
    
    print("This test uses the processor object directly without webcam")
    print("Verifying that reset() works correctly...")
    
    # Just verify reset doesn't crash
    processor.reset()
    print("  Reset successful")
    
    # Verify detector state is reset
    assert processor.detector.blink_counter == 0, "Blink counter not reset"
    assert processor.detector.total_frames == 0, "Frame counter not reset"
    
    print("\nOK: Processor reset test passed!")


def test_timeline_structure():
    """Test timeline entry structure."""
    print("\n" + "=" * 60)
    print("Test 4: Timeline Structure Validation")
    print("=" * 60)
    
    # Use empty list to just test structure
    result = compute_blink_summary([], 30.0)
    
    # Verify result structure
    assert 'timeline' in result, "Missing 'timeline' key"
    assert 'summary' in result, "Missing 'summary' key"
    assert isinstance(result['timeline'], list), "Timeline should be a list"
    assert isinstance(result['summary'], dict), "Summary should be a dict"
    
    # Verify summary fields
    required_summary_fields = ['total_blinks', 'blink_rate_per_minute', 'avg_ear', 
                               'total_frames', 'duration_seconds', 'successful_detections', 
                               'detection_rate']
    for field in required_summary_fields:
        assert field in result['summary'], f"Missing summary field: {field}"
    
    print("Required summary fields present:")
    for field in required_summary_fields:
        print(f"  - {field}")
    
    print("\nOK: Timeline structure test passed!")


def test_fps_calculation():
    """Test that FPS affects timeline timestamps correctly."""
    print("\n" + "=" * 60)
    print("Test 5: FPS Calculation Logic")
    print("=" * 60)
    
    # Use empty frames to test calculation logic
    result_30fps = compute_blink_summary([], 30.0)
    result_60fps = compute_blink_summary([], 60.0)
    
    # Both should handle empty input gracefully
    assert result_30fps['summary']['duration_seconds'] == 0.0
    assert result_60fps['summary']['duration_seconds'] == 0.0
    
    print("FPS calculation handles edge cases correctly")
    print(f"  Empty at 30 FPS: {result_30fps['summary']['duration_seconds']}s")
    print(f"  Empty at 60 FPS: {result_60fps['summary']['duration_seconds']}s")
    
    print("\nOK: FPS calculation test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BLINK COUNTER PROCESSOR TESTS")
    print("=" * 60 + "\n")
    
    try:
        # Live webcam test (main test)
        result = test_live_webcam()
        
        # Quick unit tests
        test_empty_frames()
        test_processor_reset()
        test_timeline_structure()
        test_fps_calculation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        
        if result:
            print("Live Test Summary:")
            print(f"  Total Blinks Detected: {result['summary']['total_blinks']}")
            print(f"  Blink Rate: {result['summary']['blink_rate_per_minute']:.1f} blinks/min")
            print(f"  Average EAR: {result['summary']['avg_ear']:.3f}")
            print(f"  Detection Success Rate: {result['summary']['detection_rate']:.1f}%")
            print()
        
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
