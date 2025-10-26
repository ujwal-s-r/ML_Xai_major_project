# L2CS-Net Gaze Estimation Test

## Purpose
Test L2CS-Net gaze estimation model before integrating into the video analysis pipeline.

## Installation

### Option 1: Install from GitHub (Recommended)
```powershell
# Activate your venv first
.\.venv\Scripts\Activate.ps1

# Install L2CS-Net
pip install git+https://github.com/Ahmednull/L2CS-Net.git

# Install dependencies if not already installed
pip install opencv-python torch torchvision numpy pillow
```

### Option 2: Manual Installation
If the GitHub install fails, you can try:
```powershell
pip install l2cs
```

## Running the Test

```powershell
# Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# Run the test
python tests\test_l2cs_gaze.py
```

## What the Test Does

1. **Installation Check**: Verifies L2CS package is installed
2. **Model Loading**: Loads the pre-trained L2CS-Net model (downloads weights on first run)
3. **Webcam Test**: 
   - Opens your webcam
   - Detects faces in real-time
   - Estimates gaze direction (pitch & yaw angles)
   - Shows direction: LEFT/CENTER/RIGHT and UP/CENTER/DOWN
   - Press 'q' to quit
   - Press 's' for snapshot analysis with detailed output

## Expected Output

```
============================================================
L2CS-Net Gaze Estimation Test
============================================================

✓ L2CS package found

Initializing L2CS-Net pipeline...
✓ Model loaded successfully on CPU

Testing webcam gaze estimation...
Press 'q' to quit, 's' to take snapshot test

Face 1: Pitch=-5.23°, Yaw=12.45° (CENTER-CENTER)
Face 1: Pitch=-8.12°, Yaw=-18.34° (CENTER-LEFT)
...

✅ All tests passed! L2CS-Net is working correctly.
============================================================
```

## Understanding Gaze Angles

- **Yaw** (Horizontal direction):
  - Negative (-) = Looking LEFT
  - Positive (+) = Looking RIGHT
  - Range: typically -90° to +90°
  
- **Pitch** (Vertical direction):
  - Negative (-) = Looking UP
  - Positive (+) = Looking DOWN
  - Range: typically -90° to +90°

## Thresholds for Video Analysis

For determining if user is looking at left/right parts of the video:
- **LEFT**: yaw < -15°
- **CENTER**: -15° ≤ yaw ≤ 15°
- **RIGHT**: yaw > 15°

You can adjust these thresholds based on test results.

## Troubleshooting

### Issue: "L2CS package not found"
**Solution**: Run the installation command above

### Issue: Model loading fails
**Solution**: 
- Check internet connection (weights need to be downloaded first time)
- Try running with admin privileges
- Check torch installation: `python -c "import torch; print(torch.__version__)"`

### Issue: Webcam not opening
**Solution**:
- Check if webcam is being used by another application
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`

### Issue: CUDA errors on Windows
**Solution**: The code automatically falls back to CPU if CUDA unavailable. This is fine for testing.

## Next Steps After Successful Test

1. Integrate L2CS-Net into `context/video_processor.py`
2. Create new `context/l2cs_gaze_tracker.py` wrapper
3. Update video analysis endpoint in `backend/main.py`
4. Create frontend for video recording + playback
