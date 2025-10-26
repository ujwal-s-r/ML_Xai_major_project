# L2CS-Net Integration Plan

## ✅ Test Results
- **Package Installation**: ✓ Successfully installed using `uv pip install`
- **Model Loading**: ✓ Loads from `models/L2CSNet_gaze360.pkl`
- **Webcam Test**: ✓ Running (test in progress)
- **Device**: CPU (can use CUDA if available)

## Architecture Overview

### Current Context Analyzers
1. **Emotion** (`context/emotion_analyzer.py`): FER library - 7 emotions
2. **Blink** (`context/blink_detector.py`): MediaPipe Face Mesh - EAR calculation
3. **Gaze** (`context/gaze_tracker.py` + `gaze_estimator.py`): MediaPipe - WILL BE REPLACED
4. **Iris** (`context/iris_tracker.py`): MediaPipe Iris - pupil dilation

### New Gaze System
- **Replace**: Old `gaze_tracker.py` and `gaze_estimator.py`
- **New**: `context/l2cs_gaze_tracker.py` - Wrapper around L2CS-Net Pipeline
- **Advantage**: More accurate gaze estimation, better left/right detection

## Integration Steps

### Step 1: Create L2CS Gaze Tracker Wrapper
File: `context/l2cs_gaze_tracker.py`

```python
class L2CSGazeTracker:
    def __init__(self, model_path='models/L2CSNet_gaze360.pkl'):
        # Initialize L2CS Pipeline
        # Store configuration
        
    def process_frame(self, frame):
        # Run gaze_pipeline.step(frame)
        # Extract pitch, yaw angles
        # Determine LEFT/CENTER/RIGHT region
        # Return: {gaze_direction, pitch, yaw, confidence}
        
    def get_metrics(self):
        # Return aggregated stats over video duration
        # time_looking_left, time_looking_right, time_looking_center
        # attention_score (based on center gaze time)
```

### Step 2: Update Video Processor
File: `context/video_processor.py`

Modify to use 4 analyzers:
- emotion_analyzer (keep)
- blink_detector (keep)
- **l2cs_gaze_tracker** (NEW)
- iris_tracker (keep)

Process each frame through all 4 pipelines, aggregate results.

### Step 3: Create Video Analysis Frontend
File: `frontend/video.html`

UI Components:
- Video player showing trigger video (`backend/data/video/hack.mp4`)
- Webcam feed preview (small corner overlay)
- Recording status indicator
- Progress bar (0-100%)

Flow:
1. User clicks "Start Analysis"
2. Both videos play simultaneously:
   - Large: Trigger video for user to watch
   - Small: Webcam recording (can be hidden or tiny preview)
3. WebRTC captures webcam frames
4. Frames sent to backend via WebSocket or chunked POST
5. Backend processes in real-time
6. After video ends, show summary

### Step 4: Create Backend Video Endpoint
File: `backend/main.py`

New routes:
- `GET /video` - Serve video.html
- `GET /api/video/trigger` - Stream hack.mp4 to frontend
- `POST /api/video/upload-frame` - Receive webcam frames (base64 or binary)
- `POST /api/video/submit` - Final submission with all metrics
- `WebSocket /ws/video` - Real-time frame processing (optional)

Processing flow:
1. Receive frames from frontend
2. Pass to VideoProcessor
3. Aggregate results from 4 analyzers
4. Store metrics (don't save video file)
5. Return summary: {
     emotion_timeline,
     blink_count,
     gaze_distribution: {left, center, right},
     pupil_dilation_avg
   }

### Step 5: Data Schema
File: `backend/data/video_results.jsonl`

```json
{
  "session_id": "uuid",
  "timestamp": "ISO8601",
  "duration_seconds": 60,
  "trigger_video": "hack.mp4",
  "emotion_timeline": [
    {"time": 0, "emotion": "neutral", "confidence": 0.85},
    {"time": 2, "emotion": "sad", "confidence": 0.72},
    ...
  ],
  "blink_analysis": {
    "total_blinks": 24,
    "avg_blink_rate": 0.4,  // blinks per second
    "blink_timestamps": [1.2, 3.4, 5.7, ...]
  },
  "gaze_analysis": {
    "time_looking_left": 12.3,
    "time_looking_center": 38.2,
    "time_looking_right": 9.5,
    "attention_score": 0.64,  // center time / total time
    "gaze_timeline": [
      {"time": 0, "direction": "center", "pitch": -5.2, "yaw": 2.1},
      ...
    ]
  },
  "iris_analysis": {
    "avg_pupil_size_left": 3.2,
    "avg_pupil_size_right": 3.1,
    "pupil_dilation_score": 0.15,  // change over baseline
    "pupil_timeline": [
      {"time": 0, "left": 3.0, "right": 2.9},
      ...
    ]
  }
}
```

## Technical Decisions

### Gaze Direction Thresholds
Based on L2CS-Net output (yaw angle):
- **LEFT**: yaw < -15°
- **CENTER**: -15° ≤ yaw ≤ 15°
- **RIGHT**: yaw > 15°

Can adjust based on testing.

### Processing Strategy
**Option 1: Real-time streaming** (WebSocket)
- Pros: Immediate feedback, progress bar works
- Cons: More complex, network overhead

**Option 2: Batch upload** (POST after recording)
- Pros: Simpler, more reliable
- Cons: No real-time feedback, all processing at end

**Recommendation**: Start with Option 2 for simplicity.

### Video Recording
- Use MediaRecorder API in browser
- Record to WebM format
- Send frames as base64 images OR
- Send chunks and process server-side

## Dependencies Already Installed
- ✓ opencv-python
- ✓ torch, torchvision
- ✓ l2cs
- ✓ face_detection (required by L2CS)
- ✓ numpy, scipy, pandas (from L2CS deps)

## Next Steps
1. ✅ Test L2CS-Net (IN PROGRESS)
2. Create `context/l2cs_gaze_tracker.py`
3. Update `context/video_processor.py` to use new gaze tracker
4. Test video processing pipeline locally
5. Create frontend `video.html` + `video.js`
6. Implement backend `/api/video/*` endpoints
7. Test full flow: questionnaire → game → video
8. Prepare for GenAI integration (Stage 4)
