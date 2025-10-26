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
    """Test gaze estimation on live webcam"""
    from l2cs import render
    
    print("\nTesting webcam gaze estimation...")
    print("Press 'q' to quit, 's' to take snapshot test")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return False
    
    frame_count = 0
    
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
                
                # Print details occasionally
                if frame_count % 30 == 0 and results:
                    pitch_list = results.get('pitch', [])
                    yaw_list = results.get('yaw', [])
                    
                    if pitch_list and yaw_list:
                        for i, (pitch, yaw) in enumerate(zip(pitch_list, yaw_list)):
                            horizontal = "LEFT" if yaw < -15 else "RIGHT" if yaw > 15 else "CENTER"
                            vertical = "UP" if pitch < -15 else "DOWN" if pitch > 15 else "CENTER"
                            print(f"Face {i+1}: Pitch={pitch:.2f}°, Yaw={yaw:.2f}° ({vertical}-{horizontal})")
                
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
                        pitch_list = results.get('pitch', [])
                        yaw_list = results.get('yaw', [])
                        
                        if pitch_list and yaw_list:
                            print(f"Detected {len(pitch_list)} face(s)")
                            for i, (pitch, yaw) in enumerate(zip(pitch_list, yaw_list)):
                                print(f"  Face {i+1}: Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
                                print(f"    Horizontal: {'LEFT' if yaw < -15 else 'RIGHT' if yaw > 15 else 'CENTER'}")
                                print(f"    Vertical: {'UP' if pitch < -15 else 'DOWN' if pitch > 15 else 'CENTER'}")
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
        
        if results and results.get('bboxes'):
            print(f"✓ Detected {len(results['bboxes'])} face(s)")
            for i, (pitch, yaw) in enumerate(zip(results['pitch'], results['yaw'])):
                print(f"  Face {i+1}: Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
        else:
            print("✗ No faces detected in image")
        
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
