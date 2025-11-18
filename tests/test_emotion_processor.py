"""
Test the Emotion Processor with live webcam feed.
Tests both emotion detection and XAI features (Attention Maps, Grad-CAM).
"""

import sys
from pathlib import Path
import cv2
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processors.emotion_processor import EmotionProcessor


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
    print("- Try different expressions: happy, sad, surprised, neutral")
    print("- Look directly at the camera for best face detection")
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
    print("Recording... Try different facial expressions!")
    
    frames = []
    start_time = time.time()
    frame_count = 0
    
    expressions = ["SMILE (Happy)", "FROWN (Sad)", "SURPRISED", "NEUTRAL"]
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        frames.append(frame.copy())
        frame_count += 1
        
        # Show live preview with instructions
        elapsed = time.time() - start_time
        
        # Cycle through expressions
        expr_idx = int(elapsed / (duration_seconds / len(expressions)))
        if expr_idx >= len(expressions):
            expr_idx = len(expressions) - 1
        
        instruction = f"Try: {expressions[expr_idx]}"
        color = (0, 255, 0)
        
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            instruction,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
        
        cv2.imshow('Recording - Try Different Expressions', display_frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Captured {len(frames)} frames in {time.time() - start_time:.2f}s")
    
    return frames, actual_fps


def test_live_webcam():
    """Test emotion processor with live webcam capture."""
    print("\n" + "="*60)
    print("EMOTION PROCESSOR TEST - LIVE WEBCAM")
    print("="*60)
    
    # Capture frames from webcam
    frames, fps = capture_webcam_frames(duration_seconds=10, fps=30)
    
    if not frames:
        print("\nFailed to capture frames")
        return
    
    # Initialize processor
    print("\nInitializing emotion processor...")
    processor = EmotionProcessor()
    
    # Process frames with XAI extraction
    print(f"\nProcessing {len(frames)} frames...")
    print("NOTE: XAI extraction (attention maps & Grad-CAM) may take longer...")
    result = processor.compute_emotion_summary(frames, fps, extract_xai=True)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nTIMELINE:")
    print(f"Total entries: {len(result['timeline'])}")
    
    # Show first few entries
    print("\nFirst 3 entries:")
    for i, entry in enumerate(result['timeline'][:3]):
        print(f"\n  Entry {i}:")
        print(f"    Time: {entry['t']:.2f}s")
        print(f"    Frame: {entry['frame']}")
        print(f"    Success: {entry['success']}")
        if entry['success']:
            print(f"    Dominant emotion: {entry['dominant_emotion']}")
            print(f"    Has XAI data: {entry['has_xai']}")
            if entry['emotions']:
                top_3 = sorted(entry['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top 3 emotions:")
                for emotion, prob in top_3:
                    print(f"      - {emotion}: {prob:.1f}%")
    
    # Show XAI frame examples
    xai_frames = [e for e in result['timeline'] if e['has_xai']]
    if xai_frames:
        print(f"\nXAI DATA EXAMPLES (found {len(xai_frames)} frames with XAI):")
        for i, entry in enumerate(xai_frames[:2]):  # Show first 2 XAI frames
            print(f"\n  XAI Frame {entry['frame']}:")
            print(f"    Time: {entry['t']:.2f}s")
            print(f"    Emotion: {entry['dominant_emotion']}")
            if 'attention_grid_size' in entry:
                print(f"    Attention map: {entry['attention_grid_size']}x{entry['attention_grid_size']} grid")
            if 'gradcam_target' in entry:
                print(f"    Grad-CAM target: {entry['gradcam_target']}")
    
    print("\n" + "-"*60)
    print("SUMMARY:")
    print("-"*60)
    summary = result['summary']
    
    print(f"\nTotal frames: {summary['total_frames']}")
    print(f"Duration: {summary['duration_seconds']}s")
    print(f"Successful detections: {summary['successful_detections']}")
    print(f"Detection rate: {summary['detection_rate']:.2f}%")
    
    print(f"\nDominant emotion: {summary['dominant_emotion']} (code: {summary['dominant_emotion_code']})")
    
    print(f"\nEmotion distribution:")
    for emotion, count in sorted(summary['distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / summary['total_frames'] * 100) if summary['total_frames'] > 0 else 0
        print(f"  {emotion}: {count} frames ({percentage:.1f}%)")
    
    print(f"\nXAI Analysis:")
    print(f"  XAI available: {summary['xai_available']}")
    print(f"  Frames with XAI data: {summary['xai_frame_count']}")
    if summary['xai_frames']:
        print(f"  XAI frame indices: {summary['xai_frames'][:5]}..." if len(summary['xai_frames']) > 5 else f"  XAI frame indices: {summary['xai_frames']}")
    
    # Visualize XAI on a sample frame
    if xai_frames:
        print("\n" + "-"*60)
        print("XAI VISUALIZATION")
        print("-"*60)
        print("\nGenerating XAI overlays for sample frames...")
        
        sample_frame_idx = xai_frames[0]['frame']
        sample_frame = frames[sample_frame_idx]
        
        # Extract XAI for visualization
        print(f"\nProcessing frame {sample_frame_idx} for XAI visualization...")
        attention_data = processor.extract_attention_maps(sample_frame)
        gradcam_data = processor.extract_gradcam(sample_frame)
        
        if attention_data:
            attention_overlay = processor.create_xai_overlay(sample_frame, attention_data, method='attention')
            cv2.imshow('Attention Map Overlay', attention_overlay)
            print("✓ Attention map overlay displayed")
        
        if gradcam_data:
            gradcam_overlay = processor.create_xai_overlay(sample_frame, gradcam_data, method='gradcam')
            cv2.imshow('Grad-CAM Overlay', gradcam_overlay)
            print("✓ Grad-CAM overlay displayed")
        
        if attention_data or gradcam_data:
            print("\nPress any key to close XAI visualizations...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✓ TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        test_live_webcam()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)
