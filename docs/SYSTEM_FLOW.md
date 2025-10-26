# Mental Health Assessment System - Complete Flow

## 🔄 User Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                        START: Landing Page                       │
│                     http://127.0.0.1:8000/                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 1: PHQ-8 Questionnaire                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • 8 Depression screening questions                        │  │
│  │ • 0-3 scale (Not at all → Nearly every day)             │  │
│  │ • Auto-calculates total score (0-24)                    │  │
│  │ • Determines severity level                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Data Saved: backend/data/phq8_results.jsonl                    │
│  Format: {session_id, answers[], score, severity, timestamp}    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 2: Buzzer Bombardment Game                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • 60-second reaction time game                           │  │
│  │ • Tap correct shapes (squares/circles/triangles)         │  │
│  │ • Visual & auditory distractions                         │  │
│  │ • Dynamic rule changes                                   │  │
│  │ • Tracks: accuracy, RT, impulsivity, errors             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Data Saved: backend/data/game_results.jsonl                    │
│  Format: {session_id, taps[], targets[], summary, timestamp}    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 3: Video Analysis                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Shows emotional trigger video (hack.mp4)               │  │
│  │ • Records webcam at 30 FPS                               │  │
│  │ • Real-time processing through 4 ML models:              │  │
│  │   ├─ Emotion (ViT @ 3 FPS)                              │  │
│  │   ├─ Blink Detection (MediaPipe @ 15 FPS)               │  │
│  │   ├─ Pupil Dilation (MediaPipe @ 15 FPS)                │  │
│  │   └─ Gaze Tracking (L2CS-Net @ 10 FPS)                  │  │
│  │ • Displays live stats during processing                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Data Saved: backend/data/video_results.jsonl                   │
│  Format: {session_id, timeline[], summary, timestamp}           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                  ┌───────┴───────┐
                  │               │
                  ▼               ▼
    ┌──────────────────┐  ┌──────────────────┐
    │  📊 Dashboard    │  │  🤖 AI Analysis  │
    │   (Complete)     │  │  (Coming Soon)   │
    └────────┬─────────┘  └──────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DASHBOARD VISUALIZATIONS                     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  PHQ-8 Section                                             │ │
│  │  • Total Score Card                                        │ │
│  │  • Severity Level Card                                     │ │
│  │  • Bar Chart: Individual Question Responses               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Game Performance Section                                  │ │
│  │  • Accuracy Card                                           │ │
│  │  • Reaction Time Card                                      │ │
│  │  • Impulsivity Card                                        │ │
│  │  • Errors Card                                             │ │
│  │  • Distraction Impact Cards (pre/post comparison)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Video Analysis Section                                    │ │
│  │  • Dominant Emotion Card                                   │ │
│  │  • Blink Rate Card                                         │ │
│  │  • Attention Score Card                                    │ │
│  │  • Pupil Dilations Card                                    │ │
│  │                                                             │ │
│  │  📈 Time-Series Charts:                                    │ │
│  │  1. Emotion Timeline (stepped line)                        │ │
│  │  2. Blink Activity (cumulative line)                       │ │
│  │  3. Pupil Dilation (smooth line)                           │ │
│  │  4. Gaze Direction (stacked area)                          │ │
│  │  5. Emotion Distribution (doughnut)                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

## 📊 Data Architecture

```
Session ID (UUID) - Generated at PHQ-8 start
        ↓
┌───────┴────────┐
│  localStorage  │  Persists across pages
└───────┬────────┘
        │
        ├─→ PHQ-8 Page (uses session_id)
        │       ↓
        │   phq8_results.jsonl (append)
        │
        ├─→ Game Page (reads from localStorage)
        │       ↓
        │   game_results.jsonl (append)
        │
        ├─→ Video Page (reads from localStorage)
        │       ↓
        │   video_results.jsonl (append)
        │
        └─→ Dashboard (reads from localStorage)
                ↓
            GET /api/results/{session_id}
                ↓
            Reads all 3 JSONL files
                ↓
            Returns combined JSON
                ↓
            Renders all visualizations
```

## 🔌 API Endpoints

```
Frontend Routes:
├── GET  /                → index.html (PHQ-8)
├── GET  /game            → game.html (Buzzer Bombardment)
├── GET  /video           → video.html (Video Analysis)
└── GET  /dashboard       → dashboard.html (Visualizations)

Backend API Routes:
├── GET  /api/health                    → Health check
├── POST /api/phq8/submit               → Save questionnaire
├── POST /api/game/submit               → Save game data
├── POST /api/video/start-session       → Initialize analyzer
├── POST /api/video/process-frame       → Process single frame
├── GET  /api/video/trigger             → Stream video file
├── POST /api/video/submit              → Save video results
└── GET  /api/results/{session_id}      → Fetch all session data
```

## 🎯 Data Storage

```
backend/data/
├── phq8_results.jsonl      (One line per submission)
│   └── {session_id, answers, score, severity, timestamp}
│
├── game_results.jsonl      (One line per game)
│   └── {session_id, taps, targets, server_summary, timestamp}
│
└── video_results.jsonl     (One line per video session)
    └── {session_id, timeline, server_summary, timestamp}
```

## 🚀 Complete Test Workflow

```bash
# 1. Start Backend
cd backend
python main.py

# 2. Open Browser
# Navigate to: http://127.0.0.1:8000/

# 3. PHQ-8 Questionnaire
#    - Answer 8 questions
#    - Click Submit
#    - Note: Session ID stored in localStorage

# 4. Buzzer Bombardment Game
#    - Read instructions
#    - Click Start Game
#    - Play for 60 seconds
#    - Click "Proceed to Video Analysis"

# 5. Video Analysis
#    - Click "Start Analysis"
#    - Allow webcam access
#    - Watch trigger video (~30s)
#    - Real-time processing displayed
#    - Wait for completion

# 6. Choose Next Step
#    ┌─→ Click "📊 View Dashboard"
#    │   - See all visualizations
#    │   - PHQ-8 chart
#    │   - Game metrics
#    │   - Video time-series
#    │
#    └─→ Click "🤖 AI Analysis"
#        - Coming soon!
```

## 📈 ML Models Pipeline

```
Video Frame (640x480 @ 30 FPS)
        ↓
POST /api/video/process-frame
        ↓
Backend receives base64 image
        ↓
Decode to OpenCV format (BGR)
        ↓
VideoAnalyzer.process_frame()
        ↓
    ┌───┴────┬────────┬────────┐
    ▼        ▼        ▼        ▼
Emotion  Blink    Iris    Gaze
(3fps)   (15fps)  (15fps) (10fps)
  ViT    MediaPipe MediaPipe L2CS
    └───┬────┴────────┴────────┘
        ▼
Combined JSON result
        ↓
Return to frontend
        ↓
Update live stats
        ↓
Append to timeline[]
        ↓
Video ends → Submit all data
        ↓
Save to video_results.jsonl
```

## 🎨 Technology Stack

```
Frontend:
├── HTML5 (structure)
├── CSS3 (styling + gradients)
├── Vanilla JavaScript (logic)
├── Chart.js 4.4.0 (visualizations)
└── Canvas API (video frame capture)

Backend:
├── Python 3.10+
├── FastAPI (web framework)
├── Uvicorn (ASGI server)
├── OpenCV (image processing)
├── NumPy (array operations)
└── JSONL files (data persistence)

ML Models:
├── Transformers (emotion - ViT)
├── MediaPipe (blink + iris)
├── L2CS-Net (gaze estimation)
└── PyTorch (ML backend)
```

## ✅ Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| PHQ-8 Questionnaire | ✅ Complete | Working with validation |
| Game (Buzzer Bombardment) | ✅ Complete | Full distraction analysis |
| Video Analysis | ✅ Complete | Real-time 4-model processing |
| Dashboard | ✅ Complete | All visualizations working |
| AI Analysis | 🔮 Planned | LLM integration coming |

---

**Current Version**: v1.0
**Last Updated**: October 26, 2025
**Status**: Production Ready 🚀
