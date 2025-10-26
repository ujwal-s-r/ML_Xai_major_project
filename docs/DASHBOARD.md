# Dashboard Visualization System

## Overview
The dashboard provides comprehensive visualization of all collected assessment data including PHQ-8 questionnaire, game performance, and video analysis metrics.

## Features

### 📋 PHQ-8 Depression Screening
- **Total Score**: Sum of all 8 responses (0-24 scale)
- **Severity Level**: Classification (None/Mild/Moderate/Moderately Severe/Severe)
- **Individual Responses**: Bar chart showing answers to each question
- Color-coded severity indicators

### 🎮 Game Performance Metrics
- **Accuracy**: Percentage of correct target hits
- **Reaction Time**: Average milliseconds to respond
- **Impulsivity**: Count of premature and fake target taps
- **Error Rate**: Total incorrect responses
- **Distraction Impact**: Pre/post comparison for each distraction event
  - Visual distractions (screen effects)
  - Auditory distractions (sound effects)
  - Accuracy and RT changes

### 🎥 Video Analysis Metrics

#### Summary Cards
- **Dominant Emotion**: Most frequent emotion detected
- **Blink Rate**: Blinks per minute
- **Attention Score**: Percentage of time looking at center (video)
- **Pupil Dilations**: Count of significant dilation events (>15% change)

#### Time-Series Visualizations
1. **Emotion Timeline**: Shows emotional state changes over video duration
2. **Blink Activity**: Cumulative blink count progression
3. **Pupil Dilation**: Pupil size changes (stress/arousal indicator)
4. **Gaze Direction**: Left/Center/Right distribution over time
5. **Emotion Distribution**: Pie chart of emotion percentages

## Data Flow

```
User completes video analysis
    ↓
Clicks "📊 View Dashboard"
    ↓
Dashboard loads (dashboard.html)
    ↓
JavaScript fetches: GET /api/results/{session_id}
    ↓
Backend reads JSONL files:
    ├─ phq8_results.jsonl
    ├─ game_results.jsonl
    └─ video_results.jsonl
    ↓
Returns combined JSON response
    ↓
Frontend renders:
    ├─ Summary metric cards
    ├─ Chart.js visualizations
    └─ Distraction analysis
```

## API Endpoint

### GET /api/results/{session_id}

Returns all data for a session:

```json
{
  "session_id": "uuid-here",
  "phq8": {
    "type": "phq8",
    "session_id": "uuid",
    "answers": [1, 2, 1, 0, 1, 2, 1, 1],
    "score": 9,
    "severity": "Mild",
    "timestamp": "2025-10-26T10:30:00Z",
    "version": 1
  },
  "game": {
    "type": "game",
    "session_id": "uuid",
    "duration_ms": 60000,
    "server_summary": {
      "totalEligible": 45,
      "hits": 38,
      "misses": 7,
      "errors": 5,
      "accuracy": 84,
      "avgRt": 1245,
      "impulsive": 3,
      "perDistraction": [...]
    },
    "timestamp": "2025-10-26T10:31:30Z"
  },
  "video": {
    "session_id": "uuid",
    "timestamp": "2025-10-26T10:33:00Z",
    "trigger_video": "hack.mp4",
    "duration_seconds": 30.0,
    "timeline": [...],
    "server_summary": {
      "duration_seconds": 30.0,
      "total_frames": 900,
      "emotion": {
        "distribution": {"sad": 15, "neutral": 10, "fear": 5},
        "dominant_emotion": "sad",
        "emotion_changes": 8
      },
      "blink": {
        "total_blinks": 18,
        "blink_rate_per_minute": 36.0
      },
      "pupil": {
        "avg_pupil_size": 0.0315,
        "pupil_dilation_events": 12
      },
      "gaze": {
        "distribution_percentage": {
          "left": 15.5,
          "center": 72.3,
          "right": 12.2
        },
        "attention_score": 0.723
      }
    }
  }
}
```

## Visualization Library

**Chart.js v4.4.0** (loaded via CDN)
- Lightweight and responsive
- Supports all chart types needed:
  - Bar charts (PHQ-8 responses)
  - Line charts (time-series data)
  - Doughnut chart (emotion distribution)
  - Stepped line charts (discrete state changes)

## Color Coding

### Severity Levels
- 🟢 **Success/Green**: Good performance, low severity
- 🔵 **Info/Blue**: Moderate, informational
- 🟠 **Warning/Orange**: Concerning, needs attention

### Metric Cards
- **Gradient backgrounds** for visual appeal
- **Large font sizes** for key metrics
- **Subtitles** for context

## Responsive Design

- Desktop: Multi-column grid layout
- Tablet: 2-column layout
- Mobile: Single column stacked layout
- All charts maintain aspect ratio and responsiveness

## Chart Configurations

### Emotion Timeline
- Type: Stepped line chart
- X-axis: Time in seconds
- Y-axis: Emotion categories (1-7 mapping)
- Shows confidence on hover

### Blink Timeline
- Type: Area chart (filled line)
- Shows cumulative blink count
- Smooth curve for trend

### Pupil Timeline
- Type: Smooth line chart
- Shows normalized pupil size
- Highlights dilation events

### Gaze Timeline
- Type: Stacked area chart
- 3 datasets (left/center/right)
- Binary values (0 or 1) at each timestamp

## Error Handling

1. **No Session ID**: Redirects to error message
2. **No Data Found**: 404 error with helpful message
3. **Missing Sections**: Shows "No data available" for missing components
4. **Loading States**: Spinner animation during fetch

## Future Enhancements

- [ ] Export to PDF report
- [ ] Comparison with normative data
- [ ] Trend analysis across multiple sessions
- [ ] Downloadable CSV data
- [ ] Print-optimized layout
- [ ] Share dashboard link
- [ ] Real-time updates during video analysis
- [ ] Advanced filters and date ranges

## Testing

To test the dashboard:

1. Complete full assessment flow:
   - PHQ-8 questionnaire
   - Buzzer Bombardment game
   - Video analysis

2. Click "📊 View Dashboard" button

3. Verify:
   - All metrics display correctly
   - Charts render properly
   - Time-series data shows progression
   - Distraction cards show impact

4. Check browser console for any errors

## Dependencies

- **Chart.js**: 4.4.0 (CDN)
- **Backend**: FastAPI with JSONL file reading
- **Frontend**: Vanilla JavaScript (no framework needed)
- **Storage**: Session-based localStorage for session_id

## Performance

- **Load Time**: <2 seconds for typical session
- **Chart Rendering**: <500ms per chart
- **Data Size**: ~50-200KB per session (depending on video length)
- **Browser Compatibility**: Modern browsers (Chrome, Firefox, Safari, Edge)
