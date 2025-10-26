"""
Test script for MediaPipe Blink Detection
Tests Eye Aspect Ratio (EAR) calculation and blink counting
"""

import sys
import os
# Add parent directory to path to import context modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
from context.blink_detector import BlinkDetector

def test_blink_detector():
    """Test blink detection on live webcam"""
    print("=" * 60)
    print("MediaPipe Blink Detection Test")
    print("=" * 60)
    
    # Initialize blink detector
    print("\nInitializing BlinkDetector...")
    try:
        detector = BlinkDetector()
        print("✓ BlinkDetector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return False
    
    print("✓ Webcam opened successfully")
    print("\nInstructions:")
    print("- Blink naturally and watch the counter increase")
    print("- The green bar shows Eye Aspect Ratio (EAR)")
    print("- When EAR drops below red line, a blink is detected")
    print("- Press 'q' to quit")
    print("- Press 's' for snapshot analysis")
    print("\nStarting detection...\n")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to grab frame")
            break
        
        frame_count += 1
        process_start = time.time()
        
        try:
            # Process frame for blink detection
            result = detector.detect_blink(frame)
            
            if result['success']:
                # Get EAR values
                left_ear = result.get('left_ear', 0)
                right_ear = result.get('right_ear', 0)
                avg_ear = result.get('avg_ear', 0)
                is_blinking = result.get('is_blinking', False)
                total_blinks = result.get('blink_count', 0)
                
                # Draw EAR visualization
                h, w = frame.shape[:2]
                
                # Draw EAR bars
                bar_width = 200
                bar_height = 20
                bar_x = 10
                bar_y = 50
                
                # Draw background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (50, 50, 50), -1)
                
                # Draw EAR level (normalized to 0-1, typically EAR is 0.2-0.4)
                ear_normalized = min(avg_ear / 0.4, 1.0)
                ear_bar_width = int(bar_width * ear_normalized)
                color = (0, 255, 0) if avg_ear > detector.EAR_THRESHOLD else (0, 0, 255)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + ear_bar_width, bar_y + bar_height), 
                             color, -1)
                
                # Draw threshold line
                threshold_x = int(bar_x + bar_width * (detector.EAR_THRESHOLD / 0.4))
                cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), 
                        (0, 0, 255), 2)
                
                # Display text info
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (bar_x, bar_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Blinks: {total_blinks}", (bar_x, bar_y + bar_height + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if is_blinking:
                    cv2.putText(frame, "BLINK!", (bar_x + bar_width + 20, bar_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Draw eye landmarks
                if result.get('landmarks'):
                    landmarks = result['landmarks'].landmark
                    
                    # Draw left eye
                    for idx in detector.LEFT_EYE_INDICES:
                        point = landmarks[idx]
                        x = int(point.x * w)
                        y = int(point.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Draw right eye
                    for idx in detector.RIGHT_EYE_INDICES:
                        point = landmarks[idx]
                        x = int(point.x * w)
                        y = int(point.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Print stats every 2 seconds
                if frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    blink_rate = total_blinks / elapsed if elapsed > 0 else 0
                    print(f"Time: {elapsed:.1f}s | Blinks: {total_blinks} | "
                          f"Rate: {blink_rate:.2f}/s | Avg EAR: {avg_ear:.3f}")
            
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # FPS counter
            fps = 1.0 / (time.time() - process_start) if time.time() - process_start > 0 else 0
            cv2.putText(frame, f'FPS: {fps:.1f}', (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
        
        # Display frame
        cv2.imshow('Blink Detection Test (Press Q to quit)', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✓ Test completed")
            break
        elif key == ord('s'):
            # Snapshot analysis
            print("\n--- Snapshot Analysis ---")
            result = detector.get_blink_statistics()
            print(f"Total Blinks: {result.get('total_blinks', 0)}")
            print(f"Blink Rate: {result.get('blink_rate', 0):.2f} blinks/sec")
            print(f"Average EAR: {result.get('avg_ear', 0):.3f}")
            print(f"Processing Time: {time.time() - start_time:.1f}s")
            print("------------------------\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    stats = detector.get_blink_statistics()
    print(f"Total Blinks: {stats.get('total_blinks', 0)}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration: {time.time() - start_time:.1f}s")
    print(f"Blink Rate: {stats.get('blink_rate', 0):.2f} blinks/second")
    print(f"Average EAR: {stats.get('avg_ear', 0):.3f}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_blink_detector()
        if success:
            print("\n✅ Blink detection test completed successfully!")
        else:
            print("\n❌ Blink detection test failed!")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
