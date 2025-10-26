"""
Test script for MediaPipe Iris Tracking and Pupil Dilation
Tests pupil size detection and dilation measurement
"""

import sys
import os
# Add parent directory to path to import context modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import numpy as np
from processors.iris_tracker import IrisTracker

def test_iris_tracker():
    """Test iris tracking and pupil dilation on live webcam"""
    print("=" * 60)
    print("MediaPipe Iris Tracking & Pupil Dilation Test")
    print("=" * 60)
    
    # Initialize iris tracker
    print("\nInitializing IrisTracker...")
    try:
        tracker = IrisTracker()
        print("✓ IrisTracker initialized successfully")
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
    print("- Look at the camera for pupil size measurement")
    print("- First 2 seconds: Baseline measurement (stay calm)")
    print("- After baseline: Normal viewing (pupil changes tracked)")
    print("- Green circles = iris landmarks")
    print("- Yellow circles = pupil center")
    print("- Press 'q' to quit")
    print("- Press 's' for snapshot analysis")
    print("- Press 'r' to reset baseline")
    print("\nStarting detection...\n")
    
    frame_count = 0
    start_time = time.time()
    baseline_phase = True
    baseline_end_time = start_time + 2.0  # 2 seconds for baseline
    
    pupil_size_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to grab frame")
            break
        
        frame_count += 1
        current_time = time.time()
        process_start = current_time
        
        # Check if baseline phase is over
        if baseline_phase and current_time > baseline_end_time:
            baseline_phase = False
            print("\n✓ Baseline measurement complete!")
            print("Now tracking pupil dilation changes...\n")
        
        try:
            # Process frame for iris tracking
            result = tracker.detect_iris(frame)
            
            if result['success']:
                h, w = frame.shape[:2]
                
                # Get pupil sizes
                left_pupil_size = result.get('left_pupil_size')
                right_pupil_size = result.get('right_pupil_size')
                avg_pupil_size = (left_pupil_size + right_pupil_size) / 2 if left_pupil_size and right_pupil_size else 0
                
                # Store in history
                if avg_pupil_size > 0:
                    pupil_size_history.append(avg_pupil_size)
                    if len(pupil_size_history) > 300:  # Keep last 10 seconds at 30fps
                        pupil_size_history.pop(0)
                
                # Draw iris landmarks
                if result.get('landmarks'):
                    landmarks = result['landmarks'].landmark
                    
                    # Draw left iris
                    for idx in tracker.LEFT_IRIS_LANDMARKS[:5]:  # First 5 are iris center points
                        point = landmarks[idx]
                        x = int(point.x * w)
                        y = int(point.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Draw right iris
                    for idx in tracker.RIGHT_IRIS_LANDMARKS[:5]:
                        point = landmarks[idx]
                        x = int(point.x * w)
                        y = int(point.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Draw iris centers
                    left_center = result.get('left_iris_center')
                    if left_center is not None:
                        lx, ly = left_center
                        cv2.circle(frame, (int(lx * w), int(ly * h)), 5, (0, 255, 255), -1)
                    
                    right_center = result.get('right_iris_center')
                    if right_center is not None:
                        rx, ry = right_center
                        cv2.circle(frame, (int(rx * w), int(ry * h)), 5, (0, 255, 255), -1)
                
                # Display pupil information
                y_offset = 30
                
                if baseline_phase:
                    cv2.putText(frame, "BASELINE PHASE", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    remaining = baseline_end_time - current_time
                    cv2.putText(frame, f"{remaining:.1f}s remaining", (10, y_offset + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    y_offset += 60
                
                if left_pupil_size:
                    cv2.putText(frame, f"Left Pupil: {left_pupil_size:.4f}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                
                if right_pupil_size:
                    cv2.putText(frame, f"Right Pupil: {right_pupil_size:.4f}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                
                if avg_pupil_size > 0:
                    cv2.putText(frame, f"Avg Pupil: {avg_pupil_size:.4f}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 25
                
                # Show pupil dilation ratio if available
                dilation_ratio = result.get('pupil_dilation_ratio')
                if dilation_ratio is not None:
                    color = (0, 255, 0) if abs(dilation_ratio) < 0.1 else (0, 165, 255) if abs(dilation_ratio) < 0.2 else (0, 0, 255)
                    cv2.putText(frame, f"Dilation: {dilation_ratio:+.2%}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
                
                # Draw pupil size graph (mini timeline)
                if len(pupil_size_history) > 1 and not baseline_phase:
                    graph_h = 80
                    graph_w = 200
                    graph_x = w - graph_w - 10
                    graph_y = 50
                    
                    # Draw background
                    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), 
                                 (30, 30, 30), -1)
                    
                    # Normalize and draw line
                    min_val = min(pupil_size_history)
                    max_val = max(pupil_size_history)
                    range_val = max_val - min_val if max_val > min_val else 1
                    
                    points = []
                    for i, val in enumerate(pupil_size_history):
                        x = graph_x + int((i / len(pupil_size_history)) * graph_w)
                        y = graph_y + graph_h - int(((val - min_val) / range_val) * graph_h)
                        points.append((x, y))
                    
                    # Draw line
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i + 1], (0, 255, 255), 2)
                    
                    # Labels
                    cv2.putText(frame, "Pupil Size", (graph_x, graph_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Print stats every 2 seconds
                if frame_count % 60 == 0:
                    elapsed = current_time - start_time
                    print(f"Time: {elapsed:.1f}s | Avg Pupil: {avg_pupil_size:.4f} | "
                          f"Dilation: {dilation_ratio:+.2%}" if dilation_ratio else 
                          f"Time: {elapsed:.1f}s | Avg Pupil: {avg_pupil_size:.4f}")
            
            else:
                cv2.putText(frame, "No face/iris detected", (10, 30), 
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
        cv2.imshow('Iris Tracking Test (Press Q to quit)', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✓ Test completed")
            break
        elif key == ord('s'):
            # Snapshot analysis
            print("\n--- Snapshot Analysis ---")
            stats = tracker.get_metrics()
            print(f"Average Pupil Size: {stats.get('avg_pupil_size', 0):.4f}")
            print(f"Max Pupil Size: {stats.get('max_pupil_size', 0):.4f}")
            print(f"Min Pupil Size: {stats.get('min_pupil_size', 0):.4f}")
            print(f"Pupil Dilation Delta: {stats.get('pupil_dilation_delta', 0):.4f}")
            print(f"Baseline: {stats.get('baseline_pupil_size', 0):.4f}")
            print("------------------------\n")
        elif key == ord('r'):
            # Reset baseline
            print("\nResetting baseline...")
            tracker.reset()
            baseline_phase = True
            baseline_end_time = time.time() + 2.0
            pupil_size_history.clear()
            print("Baseline reset. Collecting new baseline for 2 seconds...\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    stats = tracker.get_metrics()
    print(f"Average Pupil Size: {stats.get('avg_pupil_size', 0):.4f}")
    print(f"Max Pupil Size: {stats.get('max_pupil_size', 0):.4f}")
    print(f"Min Pupil Size: {stats.get('min_pupil_size', 0):.4f}")
    print(f"Pupil Size Range: {stats.get('max_pupil_size', 0) - stats.get('min_pupil_size', 0):.4f}")
    print(f"Dilation Delta: {stats.get('pupil_dilation_delta', 0):.4f}")
    print(f"Baseline Size: {stats.get('baseline_pupil_size', 0):.4f}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration: {time.time() - start_time:.1f}s")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_iris_tracker()
        if success:
            print("\n✅ Iris tracking test completed successfully!")
        else:
            print("\n❌ Iris tracking test failed!")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
