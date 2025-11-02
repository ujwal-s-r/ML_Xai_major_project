"""
Test script for ViT-based Emotion Detection
Uses the same EmotionAnalyzer as the application (processors/emotion_analyzer.py)

Features:
- Live webcam test with on-screen visualization
- Snapshot logging of dominant emotion and top probabilities
- Prints FPS and device info

Keys:
- q: quit
- s: snapshot summary to console
"""

import sys
import os
# Add parent directory to path to import processors modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
from processors.emotion_analyzer import EmotionAnalyzer


def test_emotion_detection_webcam():
    print("=" * 60)
    print("ViT Emotion Detection Test (Webcam)")
    print("=" * 60)

    # Initialize analyzer
    print("\nInitializing EmotionAnalyzer (ViT)...")
    try:
        analyzer = EmotionAnalyzer()
        device = analyzer.device if hasattr(analyzer, 'device') else 'cpu'
        print(f"✓ EmotionAnalyzer initialized on device: {device}")
    except Exception as e:
        print(f"✗ Failed to initialize EmotionAnalyzer: {e}")
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
    print("- Keep your face centered and well lit")
    print("- The top shows the dominant emotion and whether its result is cached")
    print("- The list shows emotion probabilities from the ViT model")
    print("- Press 's' for snapshot summary; 'q' to quit")
    print("\nStarting detection...\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to grab frame")
                break

            frame_count += 1
            process_start = time.time()

            try:
                result = analyzer.detect_emotion(frame)

                # Draw visualization using analyzer's helper for consistency
                vis_frame = analyzer.visualize_emotion_detection(frame, result)

                # Compute FPS
                fps = 1.0 / (time.time() - process_start) if time.time() - process_start > 0 else 0
                cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show
                cv2.imshow('ViT Emotion Detection Test (Press Q to quit, S to snapshot)', vis_frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✓ Test completed")
                    break
                elif key == ord('s'):
                    # Snapshot summary to console
                    print("\n--- Snapshot Analysis ---")
                    if result and result.get('success'):
                        dom = result.get('dominant_emotion')
                        emotions = result.get('emotions', {})
                        print(f"Dominant: {dom}")
                        # Top 5 emotions
                        for k, v in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]:
                            print(f"  {k:10s}: {v:.1f}%")
                    else:
                        print("No face detected or model failed on this frame")
                    elapsed = time.time() - start_time
                    print(f"Frames processed: {frame_count} | Elapsed: {elapsed:.1f}s")
                    print("------------------------\n")

            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                # Continue loop

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"Total Frames: {frame_count}")
    print(f"Duration: {elapsed:.1f}s")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_emotion_detection_webcam()
        if success:
            print("\n✅ Emotion detection test completed successfully!")
        else:
            print("\n❌ Emotion detection test failed!")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
