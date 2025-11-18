"""
Real-time blink detection test with live visualization.
This helps debug and calibrate the blink detector.
"""

import sys
from pathlib import Path
import cv2
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from context.blink_detector import BlinkDetector


def test_realtime_blink_detection(duration_seconds=30):
    """
    Test blink detection in real-time with live visualization.
    Shows EAR values and blink detection status.
    """
    print("\n" + "="*60)
    print("REAL-TIME BLINK DETECTION TEST")
    print("="*60)
    print("\nInstructions:")
    print("- Look at the camera")
    print("- Blink naturally a few times")
    print("- Watch the EAR values (should be ~0.25-0.35 when open)")
    print("- Watch for 'BLINK DETECTED!' messages")
    print("- Press 'q' to quit early")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Initialize blink detector
    detector = BlinkDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector.fps = fps
    
    print(f"\n✓ Webcam opened at {fps} FPS")
    print("\nRecording... (Press 'q' to stop)\n")
    
    start_time = time.time()
    frame_count = 0
    last_blink_count = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Detect blink
            blink_data = detector.detect_blink(frame)
            
            # Check if new blink detected
            if blink_data['blink_count'] > last_blink_count:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.2f}s] ✓ BLINK DETECTED! (Total: {blink_data['blink_count']})")
                last_blink_count = blink_data['blink_count']
            
            # Visualize on frame
            output_frame = detector.visualize_blink_detection(frame, blink_data)
            
            # Add EAR value display
            if blink_data['success']:
                # Color code: green when eyes open, red when blinking
                color = (0, 0, 255) if blink_data['is_blinking'] else (0, 255, 0)
                
                # Display EAR values
                cv2.putText(
                    output_frame,
                    f"EAR: {blink_data['avg_ear']:.3f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                
                # Display left/right EAR
                cv2.putText(
                    output_frame,
                    f"L: {blink_data['left_ear']:.3f} R: {blink_data['right_ear']:.3f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
                
                # Display blink status
                status = "BLINKING!" if blink_data['is_blinking'] else "Eyes Open"
                cv2.putText(
                    output_frame,
                    status,
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                
                # Display blink count
                cv2.putText(
                    output_frame,
                    f"Blinks: {blink_data['blink_count']}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
            
            # Display time remaining
            remaining = duration_seconds - (time.time() - start_time)
            cv2.putText(
                output_frame,
                f"Time: {remaining:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow('Blink Detection Test - Press Q to quit', output_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Duration: {elapsed:.2f}s")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / elapsed:.2f}")
    print(f"Total blinks detected: {last_blink_count}")
    print(f"Blink rate: {(last_blink_count / elapsed) * 60:.2f} blinks/minute")
    print(f"Normal blink rate: 15-30 blinks/minute")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        test_realtime_blink_detection(duration_seconds=30)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
