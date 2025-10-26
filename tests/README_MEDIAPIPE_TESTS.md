# MediaPipe Tests - Blink & Iris Tracking

## Overview
Tests for MediaPipe-based facial analysis components before integrating into video processing pipeline.

## Tests Available

### 1. Blink Detection Test (`test_blink_detection.py`)
Tests Eye Aspect Ratio (EAR) calculation and blink counting using MediaPipe Face Mesh.

**What it does:**
- Detects eye landmarks using MediaPipe Face Mesh
- Calculates Eye Aspect Ratio (EAR) for both eyes
- Counts blinks when EAR drops below threshold
- Visualizes EAR in real-time with green bar
- Shows blink counter and detection status

**Running the test:**
```powershell
python tests\test_blink_detection.py
```

**Controls:**
- **q**: Quit test
- **s**: Show snapshot analysis with statistics
- Just blink naturally and watch the counter!

**What to look for:**
- ✅ EAR value around 0.25-0.35 when eyes open
- ✅ EAR drops below 0.20 (red line) when blinking
- ✅ Blink counter increments correctly
- ✅ Green dots on eye landmarks
- ✅ No lag or crashes

**Expected Output:**
```
============================================================
MediaPipe Blink Detection Test
============================================================
✓ BlinkDetector initialized successfully
✓ Webcam opened successfully
...
Time: 10.0s | Blinks: 12 | Rate: 1.20/s | Avg EAR: 0.312
```

---

### 2. Iris Tracking Test (`test_iris_tracking.py`)
Tests pupil size measurement and dilation tracking using MediaPipe Iris.

**What it does:**
- Detects iris landmarks using MediaPipe Face Mesh with iris refinement
- Measures pupil size for both eyes
- Establishes baseline pupil size (first 2 seconds)
- Tracks pupil dilation changes over time
- Visualizes pupil size history in mini-graph

**Running the test:**
```powershell
python tests\test_iris_tracking.py
```

**Controls:**
- **q**: Quit test
- **s**: Show snapshot analysis with pupil statistics
- **r**: Reset baseline measurement
- First 2 seconds: Stay calm for baseline measurement

**What to look for:**
- ✅ Baseline measurement completes after 2 seconds
- ✅ Pupil size values displayed (typically 0.02-0.05)
- ✅ Dilation ratio shows percentage change from baseline
- ✅ Mini-graph shows pupil size timeline
- ✅ Green dots on iris, yellow dots at pupil center

**Expected Output:**
```
============================================================
MediaPipe Iris Tracking & Pupil Dilation Test
============================================================
✓ IrisTracker initialized successfully
✓ Webcam opened successfully
...
✓ Baseline measurement complete!
Time: 10.0s | Avg Pupil: 0.0342 | Dilation: +5.23%
```

**Understanding Dilation:**
- **Baseline (2s)**: Initial calm state pupil size
- **Positive %**: Pupil dilated (larger than baseline) - arousal, stress, interest
- **Negative %**: Pupil constricted (smaller) - calm, relaxed
- **Typical range**: ±10-20% for normal viewing

---

## Dependencies Installed

All required packages are now in the venv:
- ✅ `mediapipe==0.10.21` - Face mesh and iris tracking
- ✅ `opencv-python==4.12.0.88` - Video capture and display
- ✅ `numpy==1.26.4` - Array operations
- ✅ `transformers==4.57.1` - For emotion analysis (next stage)

## Integration Status

| Component | Test Status | Integration Status | Model |
|-----------|-------------|-------------------|-------|
| **Gaze (L2CS-Net)** | ✅ Tested | 🔜 Pending | L2CS-Net ResNet50 |
| **Blink Detection** | 🧪 Testing | 🔜 Pending | MediaPipe Face Mesh |
| **Iris Tracking** | ⏳ Next | 🔜 Pending | MediaPipe Iris |
| **Emotion Analysis** | ⏳ TODO | 🔜 Pending | Transformers ViT |

## Next Steps

1. ✅ Test L2CS-Net gaze estimation
2. 🧪 Test MediaPipe blink detection (current)
3. ⏳ Test MediaPipe iris tracking
4. 🔜 Create unified video processor
5. 🔜 Build video analysis frontend
6. 🔜 Implement backend video endpoints
7. 🔜 Test full pipeline with trigger video

## Troubleshooting

### Issue: "No module named 'mediapipe'"
**Solution:** Run `uv pip install mediapipe`

### Issue: "No module named 'transformers'"
**Solution:** Run `uv pip install transformers`

### Issue: Webcam not opening
**Solution:** 
- Close other applications using webcam
- Try different camera ID: change `cv2.VideoCapture(0)` to `VideoCapture(1)`

### Issue: Low FPS / Laggy
**Solution:**
- MediaPipe can be CPU-intensive
- This is normal for testing, will optimize for production
- GPU acceleration available if CUDA installed

### Issue: No face detected
**Solution:**
- Ensure good lighting
- Face the camera directly
- Move closer to camera
- Check if webcam is working in other apps

## Model Details

### MediaPipe Face Mesh
- **Landmarks:** 478 facial landmarks (including iris landmarks 468-477)
- **Iris landmarks:** 5 points per eye for precise pupil tracking
- **Accuracy:** High precision for facial features
- **Speed:** Real-time capable on CPU

### Eye Aspect Ratio (EAR)
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```
- **Open eye:** EAR ≈ 0.25-0.35
- **Closed eye:** EAR < 0.20
- **Threshold:** 0.20 (configurable)

### Pupil Dilation Measurement
- Uses iris landmarks to calculate pupil diameter
- Normalized to account for face distance
- Baseline established in first 2 seconds
- Dilation expressed as % change from baseline
