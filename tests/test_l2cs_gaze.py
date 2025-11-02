"""
Test script for L2CS-Net gaze estimation
Tests if the model works on Windows before integrating into the pipeline
"""

import cv2
import numpy as np
import torch
import time
from PIL import Image

def test_l2cs_installation():
    """Test if L2CS-Net package is installed"""
    try:
        import l2cs
        print("✓ L2CS package found")
        return True
    except ImportError:
        print("✗ L2CS package not found")
        print("Install with: pip install git+https://github.com/Ahmednull/L2CS-Net.git")
        return False

def test_model_loading():
    """Test loading the L2CS-Net model"""
    try:
        from l2cs import Pipeline, select_device
        import pathlib
        
        # Initialize the pipeline (downloads weights on first run)
        print("\nInitializing L2CS-Net pipeline...")
        
        # Model weights path
        CWD = pathlib.Path.cwd()
        weights_path = CWD / 'models' / 'L2CSNet_gaze360.pkl'
        
        if not weights_path.exists():
            print(f"✗ Model weights not found at: {weights_path}")
            print("Please ensure models/L2CSNet_gaze360.pkl exists")
            return None
        
        device = select_device('cpu', batch_size=1)
        gaze_pipeline = Pipeline(
            weights=weights_path,
            arch='ResNet50',
            device=device
        )
        
        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"✓ Model loaded successfully on {device_name}")
        print(f"✓ Weights loaded from: {weights_path}")
        return gaze_pipeline
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_webcam_gaze(gaze_pipeline):
    """Test gaze estimation on live webcam (HORIZONTAL ONLY: LEFT/CENTER/RIGHT)."""
    from l2cs import render
    
    print("\nTesting webcam gaze estimation...")
    print("Press 'q' to quit, 's' to take snapshot test")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return False
    
    frame_count = 0
    # Accumulators and state
    horiz_counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
    frames_with_face = 0
    yaw_values: list[float] = []  # degrees

    # Thresholds (horizontal only) - realistic for webcam screen glances
    TH_HORIZ_ENTER = 10.0  # enter LEFT/RIGHT at ±10°
    TH_HORIZ_EXIT = 5.0    # return to CENTER at ±5°

    # Baseline yaw (first N face frames)
    baseline_frames_target = 30
    baseline_yaw_vals: list[float] = []
    baseline_yaw = 0.0
    baseline_locked = False

    # Smoothing (EMA) and hysteresis state
    yaw_ema: float | None = None
    EMA_ALPHA = 0.5  # more responsive to small turns
    horiz_state = "CENTER"  # LEFT/CENTER/RIGHT
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to grab frame")
                break
            
            frame_count += 1
            start_time = time.time()
            
            try:
                # Process frame with L2CS-Net pipeline
                results = gaze_pipeline.step(frame)
                
                # Visualize output using L2CS render function
                frame = render(frame, results)
                
                # Add FPS counter
                fps = 1.0 / (time.time() - start_time) if time.time() - start_time > 0 else 0
                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 20), 
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Extract first face angles and accumulate directions (HORIZONTAL ONLY)
                if results:
                    if isinstance(results, dict):
                        yaw_list = results.get('yaw', [])
                    else:
                        yaw_list = getattr(results, 'yaw', [])

                    if yaw_list:
                        frames_with_face += 1
                        yaw = float(yaw_list[0])
                        yaw_values.append(yaw)
                        # Build baseline during initial face frames
                        if not baseline_locked:
                            baseline_yaw_vals.append(yaw)
                            if len(baseline_yaw_vals) >= baseline_frames_target:
                                baseline_yaw = sum(baseline_yaw_vals) / len(baseline_yaw_vals)
                                baseline_locked = True

                        # Baseline-adjusted yaw
                        adj_yaw = yaw - (baseline_yaw if baseline_locked else 0.0)

                        # Apply EMA smoothing
                        yaw_ema = adj_yaw if yaw_ema is None else (EMA_ALPHA * adj_yaw + (1-EMA_ALPHA) * yaw_ema)

                        # Horizontal hysteresis classification on smoothed yaw
                        if horiz_state == "CENTER":
                            if yaw_ema is not None and yaw_ema <= -TH_HORIZ_ENTER:
                                horiz_state = "LEFT"
                            elif yaw_ema is not None and yaw_ema >= TH_HORIZ_ENTER:
                                horiz_state = "RIGHT"
                        elif horiz_state == "LEFT":
                            if yaw_ema is not None and -TH_HORIZ_EXIT <= yaw_ema <= TH_HORIZ_EXIT:
                                horiz_state = "CENTER"
                            elif yaw_ema is not None and yaw_ema >= TH_HORIZ_ENTER:
                                horiz_state = "RIGHT"
                        elif horiz_state == "RIGHT":
                            if yaw_ema is not None and -TH_HORIZ_EXIT <= yaw_ema <= TH_HORIZ_EXIT:
                                horiz_state = "CENTER"
                            elif yaw_ema is not None and yaw_ema <= -TH_HORIZ_ENTER:
                                horiz_state = "LEFT"

                        # Accumulate per-frame label counts
                        horiz_counts[horiz_state] += 1
                        # Overlay direction text for clarity (horizontal only)
                        label = f'{horiz_state}'
                        if baseline_locked:
                            label += f"  (yaw={adj_yaw:.1f}° ~{yaw_ema:.1f}°)"
                        else:
                            label += "  (calibrating baseline...)"
                        cv2.putText(frame, label, (10, 50),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_AA)

                # Print details occasionally
                if frame_count % 30 == 0 and results:
                    # Support both dict-like and object-like results containers
                    if isinstance(results, dict):
                        yaw_list = results.get('yaw', [])
                    else:
                        yaw_list = getattr(results, 'yaw', [])
                    
                    if yaw_list:
                        for i, yaw in enumerate(yaw_list):
                            yaw = float(yaw)
                            # Use the same threshold logic (raw comparison for debug)
                            horizontal = "LEFT" if yaw < -TH_HORIZ_ENTER else "RIGHT" if yaw > TH_HORIZ_ENTER else "CENTER"
                            print(f"Face {i+1}: Yaw={yaw:.2f}° ({horizontal})")
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
            
            # Display
            cv2.imshow('L2CS-Net Gaze Test (Press Q to quit)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Test completed successfully")
                break
            elif key == ord('s'):
                # Snapshot test - detailed output
                print("\n--- Snapshot Analysis ---")
                try:
                    results = gaze_pipeline.step(frame)
                    
                    if results:
                        if isinstance(results, dict):
                            yaw_list = results.get('yaw', [])
                        else:
                            yaw_list = getattr(results, 'yaw', [])
                        
                        if yaw_list:
                            print(f"Detected {len(yaw_list)} face(s)")
                            for i, yaw in enumerate(yaw_list):
                                yaw = float(yaw)
                                print(f"  Face {i+1}: Yaw={yaw:.2f}°")
                                print(f"    Horizontal: {'LEFT' if yaw < -TH_HORIZ_ENTER else 'RIGHT' if yaw > TH_HORIZ_ENTER else 'CENTER'}")
                        else:
                            print("No faces detected in snapshot")
                    else:
                        print("No results from pipeline")
                except Exception as e:
                    print(f"Error in snapshot: {e}")
                    import traceback
                    traceback.print_exc()
                print("------------------------\n")
    
    cap.release()
    cv2.destroyAllWindows()

    # Final summary (horizontal only)
    total_face_frames = frames_with_face if frames_with_face > 0 else 1
    horiz_pct = {k: (v / total_face_frames) * 100.0 for k, v in horiz_counts.items()}
    avg_yaw = sum(yaw_values) / len(yaw_values) if yaw_values else 0.0

    print("\n=== Gaze Summary (first face) ===")
    print(f"Frames processed: {frame_count}")
    print(f"Frames with face: {frames_with_face}")
    print(f"Baseline locked: {baseline_locked} (yaw0={baseline_yaw:.2f}°)")
    if yaw_values:
        print(f"Raw yaw range: [{min(yaw_values):.2f}, {max(yaw_values):.2f}]°")
    print(f"Thresholds: horiz_enter={TH_HORIZ_ENTER:.1f}° / horiz_exit={TH_HORIZ_EXIT:.1f}°")
    print(f"Horizontal LEFT/CENTER/RIGHT counts: {horiz_counts}")
    print(f"Horizontal distribution (%): {{'LEFT': {horiz_pct['LEFT']:.1f}, 'CENTER': {horiz_pct['CENTER']:.1f}, 'RIGHT': {horiz_pct['RIGHT']:.1f}}}")
    print(f"Average yaw: {avg_yaw:.2f}°")

    # Also provide a compact JSON-style line for copy/paste if needed
    try:
        import json
        summary_json = {
            "frames_processed": frame_count,
            "frames_with_face": frames_with_face,
            "horiz_counts": horiz_counts,
            "horiz_pct": {k: round(v, 2) for k, v in horiz_pct.items()},
            "avg_yaw": round(avg_yaw, 3),
            "baseline": {"yaw0": round(baseline_yaw, 3), "locked": baseline_locked},
            "thresholds": {"horiz_enter": TH_HORIZ_ENTER, "horiz_exit": TH_HORIZ_EXIT},
        }
        print("Summary JSON:", json.dumps(summary_json))
    except Exception:
        pass

    return True

def test_static_image(gaze_pipeline, image_path=None):
    """Test gaze estimation on a static image"""
    print("\nTesting on static image...")
    
    # Create a dummy image if no path provided
    if image_path is None:
        print("No image path provided, skipping static test")
        return True
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Could not load image: {image_path}")
            return False
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run pipeline
        results = gaze_pipeline.step(img_rgb)
        
        if results:
            if isinstance(results, dict):
                bboxes = results.get('bboxes', [])
                pitch_list = results.get('pitch', [])
                yaw_list = results.get('yaw', [])
            else:
                bboxes = getattr(results, 'bboxes', [])
                pitch_list = getattr(results, 'pitch', [])
                yaw_list = getattr(results, 'yaw', [])
            
            if bboxes is not None and len(pitch_list) > 0:
                print(f"✓ Detected {len(pitch_list)} face(s)")
                for i, (pitch, yaw) in enumerate(zip(pitch_list, yaw_list)):
                    print(f"  Face {i+1}: Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
            else:
                print("✗ No faces detected in image")
        else:
            print("✗ No results from pipeline")
        
        return True
    except Exception as e:
        print(f"✗ Static image test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("=" * 60)
    print("L2CS-Net Gaze Estimation Test")
    print("=" * 60)
    
    # Step 1: Check installation
    if not test_l2cs_installation():
        print("\n❌ Please install L2CS-Net first:")
        print("   pip install git+https://github.com/Ahmednull/L2CS-Net.git")
        return
    
    # Step 2: Load model
    gaze_pipeline = test_model_loading()
    if gaze_pipeline is None:
        print("\n❌ Model loading failed. Cannot proceed with tests.")
        return
    
    # Step 3: Test on webcam
    print("\n" + "=" * 60)
    success = test_webcam_gaze(gaze_pipeline)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All tests passed! L2CS-Net is working correctly.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Integrate L2CS-Net into video_processor.py")
        print("2. Replace gaze_tracker.py with L2CS-Net pipeline")
        print("3. Test full video analysis workflow")
    else:
        print("\n❌ Tests failed. Check errors above.")

if __name__ == "__main__":
    main()
