# Mental Health Assessment System

A comprehensive multi-stage mental health assessment platform combining PHQ-8 questionnaire, cognitive game testing, real-time video analysis, and data visualization.

## 🌟 Features

### Stage 1: PHQ-8 Depression Screening
- 8-question standardized depression assessment
- Auto-calculated severity scoring (0-24 scale)
- Session-based tracking with UUID

### Stage 2: Buzzer Bombardment Game
- 60-second reaction time and attention test
- Visual and auditory distractions
- Dynamic rule changes
- Comprehensive metrics: accuracy, reaction time, impulsivity, errors

### Stage 3: Real-time Video Analysis
- Webcam recording during emotional trigger video
- 4 ML models processing in parallel:
  - **Emotion Analysis** (Transformers ViT @ 3 FPS)
  - **Blink Detection** (MediaPipe @ 15 FPS)
  - **Pupil Dilation** (MediaPipe @ 15 FPS)
  - **Gaze Tracking** (L2CS-Net @ 10 FPS)
- Live stats display during processing
- Complete time-series data collection

### Stage 4: Interactive Dashboard
- PHQ-8 visualization with severity indicators
- Game performance metrics and distraction impact analysis
- Video analysis time-series charts:
  - Emotion timeline
  - Blink activity
  - Pupil dilation patterns
  - Gaze direction tracking
  - Emotion distribution
- Professional Chart.js visualizations

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS + Chart.js)
    ↓
FastAPI Backend
    ↓
ML Models (ViT, MediaPipe, L2CS-Net)
    ↓
JSONL Data Storage
```

## 📁 Project Structure

```
ML_Xai_major_project/
├── backend/
│   ├── main.py              # FastAPI application
│   └── data/                # JSONL storage
│       ├── phq8_results.jsonl
│       ├── game_results.jsonl
│       └── video_results.jsonl
├── frontend/
│   ├── index.html           # PHQ-8 questionnaire
│   ├── game.html            # Buzzer Bombardment
│   ├── video.html           # Video analysis
│   ├── dashboard.html       # Data visualization
│   └── *.js                 # Frontend logic
├── processors/              # ML model processors
│   ├── emotion_analyzer.py
│   ├── blink_detector.py
│   ├── iris_tracker.py
│   └── video_analyzer.py
├── models/                  # Model weights
│   └── L2CSNet_gaze360.pkl
├── tests/                   # Model tests
└── docs/                    # Documentation
```

## 🚀 Quick Start

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam (for video analysis)
- Modern browser (Chrome, Firefox, Safari, Edge)

### Installation

```powershell
# Clone repository
git clone https://github.com/ujwal-s-r/ML_Xai_major_project.git
cd ML_Xai_major_project

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install ML models (if not already)
uv pip install git+https://github.com/Ahmednull/L2CS-Net.git
uv pip install mediapipe transformers torch opencv-python
```

### Running the Application

```powershell
# Start backend server
cd backend
python main.py

# Or use uvicorn directly
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open browser: **http://127.0.0.1:8000**

## 🎯 Usage Flow

1. **PHQ-8 Questionnaire** (`/`)
   - Answer 8 depression screening questions
   - Submit to get severity level
   - Session ID auto-generated

2. **Buzzer Bombardment Game** (`/game`)
   - Click "Start Game"
   - Tap correct shapes for 60 seconds
   - Avoid fake targets and follow rule changes

3. **Video Analysis** (`/video`)
   - Click "Start Analysis"
   - Allow webcam access
   - Watch trigger video while being recorded
   - Wait for ML processing to complete

4. **View Dashboard** (`/dashboard`)
   - Click "📊 View Dashboard" after video
   - Explore all visualizations
   - Review time-series data

## 🔌 API Endpoints

### PHQ-8
- `POST /api/phq8/submit` - Submit questionnaire answers

### Game
- `POST /api/game/submit` - Submit game performance data

### Video
- `POST /api/video/start-session` - Initialize video analyzer
- `POST /api/video/process-frame` - Process single webcam frame
- `GET /api/video/trigger` - Stream trigger video file
- `POST /api/video/submit` - Submit complete video analysis

### Dashboard
- `GET /api/results/{session_id}` - Fetch all session data

## 📊 Data Format

All data stored as JSONL (one JSON object per line):

### PHQ-8 Results
```json
{
  "session_id": "uuid",
  "answers": [1, 2, 1, 0, 1, 2, 1, 1],
  "score": 9,
  "severity": "Mild",
  "timestamp": "2025-10-26T10:30:00Z"
}
```

### Game Results
```json
{
  "session_id": "uuid",
  "duration_ms": 60000,
  "server_summary": {
    "accuracy": 84,
    "avgRt": 1245,
    "impulsive": 3,
    "perDistraction": [...]
  },
  "timestamp": "2025-10-26T10:31:30Z"
}
```

### Video Results
```json
{
  "session_id": "uuid",
  "timeline": [...],
  "server_summary": {
    "emotion": {...},
    "blink": {...},
    "pupil": {...},
    "gaze": {...}
  },
  "timestamp": "2025-10-26T10:33:00Z"
}
```

## 🧪 Testing

### Individual Model Tests
```powershell
# Test blink detection
python tests/test_blink_detection.py

# Test iris tracking
python tests/test_iris_tracking.py

# Test gaze estimation
python tests/test_l2cs_gaze.py
```

### Complete System Test
1. Start backend: `python backend/main.py`
2. Open: http://127.0.0.1:8000
3. Complete all 3 stages
4. View dashboard

## 🛠️ Technology Stack

### Frontend
- HTML5, CSS3, Vanilla JavaScript
- Chart.js 4.4.0 (visualizations)
- Canvas API (frame capture)

### Backend
- FastAPI (web framework)
- Uvicorn (ASGI server)
- OpenCV (image processing)
- NumPy (array operations)

### ML Models
- **Emotion**: Transformers (trpakov/vit-face-expression)
- **Blink**: MediaPipe Face Mesh
- **Iris**: MediaPipe Iris
- **Gaze**: L2CS-Net (ResNet50)

### Storage
- JSONL files (append-only)
- localStorage (session management)

## 📖 Documentation

Detailed documentation in `/docs`:
- `VIDEO_PROCESSING.md` - Video analysis architecture
- `DASHBOARD.md` - Dashboard features and API
- `DASHBOARD_IMPLEMENTATION.md` - Implementation details
- `SYSTEM_FLOW.md` - Complete system flow diagram

## 🔮 Future Features

- [ ] AI Analysis with LLM integration (GPT-4/Claude)
- [ ] PDF report generation
- [ ] Multi-session comparison
- [ ] Normative data comparison
- [ ] Export to CSV
- [ ] Real-time WebSocket streaming
- [ ] GPU acceleration

## 📝 Notes

- **CORS**: Permissive for development; tighten for production
- **Privacy**: All data stored locally; no external transmission
- **Session**: UUID-based tracking across all stages
- **Performance**: Real-time processing optimized with frame sampling
- **Browser**: Requires webcam permissions for video analysis

## 🤝 Contributing

This is a research/educational project. Contributions welcome!

## 📄 License

Educational/Research Use

## 👥 Authors

- Ujwal S R ([@ujwal-s-r](https://github.com/ujwal-s-r))

## 🙏 Acknowledgments

- L2CS-Net for gaze estimation
- MediaPipe for facial landmark detection
- Hugging Face Transformers for emotion analysis
- FastAPI for excellent web framework

---

**Status**: ✅ Production Ready (v1.0)
**Last Updated**: October 26, 2025
