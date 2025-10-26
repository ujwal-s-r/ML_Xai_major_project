from __future__ import annotations

import json
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# Import video analyzer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processors.video_analyzer import VideoAnalyzer

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "backend" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = DATA_DIR / "phq8_results.jsonl"

app = FastAPI(title="PHQ-8 Questionnaire Service")

# If you plan to open the HTML from file:// or a different origin, configure CORS as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # keep permissive for now; tighten later when deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend assets
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Initialize Video Analyzer (singleton for all requests)
video_analyzer = None

def get_video_analyzer():
    """Lazy-load video analyzer to avoid startup delay."""
    global video_analyzer
    if video_analyzer is None:
        print("Initializing Video Analyzer...")
        video_analyzer = VideoAnalyzer(
            emotion_sample_rate=10,  # 3 FPS at 30fps
            blink_sample_rate=2,     # 15 FPS at 30fps
            iris_sample_rate=2,      # 15 FPS at 30fps
            gaze_sample_rate=3       # 10 FPS at 30fps
        )
    return video_analyzer


class PHQ8Submission(BaseModel):
    session_id: str
    answers: List[int]

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, v: List[int]) -> List[int]:
        if len(v) != 8:
            raise ValueError("answers must have exactly 8 integers (0-3)")
        for score in v:
            if not isinstance(score, int) or score < 0 or score > 3:
                raise ValueError("each answer must be an integer between 0 and 3 inclusive")
        return v


def phq8_severity(total: int) -> str:
    # Standard PHQ-8 severity bands
    if total <= 4:
        return "None"
    if total <= 9:
        return "Mild"
    if total <= 14:
        return "Moderate"
    if total <= 19:
        return "Moderately Severe"
    return "Severe"


def append_result(record: dict) -> None:
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/")
async def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend index not found")
    return FileResponse(str(index_path))


@app.get("/game")
async def game_page() -> FileResponse:
    game_path = FRONTEND_DIR / "game.html"
    if not game_path.exists():
        raise HTTPException(status_code=500, detail="Game page not found")
    return FileResponse(str(game_path))


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/api/phq8/submit")
async def submit_phq8(payload: PHQ8Submission):
    total = sum(payload.answers)
    severity = phq8_severity(total)
    record = {
        "type": "phq8",
        "session_id": payload.session_id,
        "answers": payload.answers,
        "score": total,
        "severity": severity,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": 1,
    }

    try:
        append_result(record)
    except Exception as e:
        # Even if persistence fails, return the score (but indicate saved=False)
        return {
            "score": total,
            "severity": severity,
            "saved": False,
            "error": str(e),
        }

    return {"score": total, "severity": severity, "saved": True}


# Convenience alias and favicon handler to reduce 404 noise
@app.get("/health")
async def health_alias():
    return await health()


@app.get("/favicon.ico")
async def favicon():
    # No favicon yet; return 204 to avoid 404 log noise
    return Response(status_code=204)


# ---- Game submission ----
class GameSubmission(BaseModel):
    session_id: str
    duration_ms: int
    taps: list
    targets: list
    distractions: list
    rule_windows: list | None = None
    summary: dict | None = None
    timestamp: str | None = None
    version: int | None = 1


GAME_RESULTS_PATH = DATA_DIR / "game_results.jsonl"


def _rule_at(ts: float, windows: list[dict]) -> str:
    # windows: list of {rule, startTs, endTs}
    if not windows:
        return "normal"
    for w in windows:
        st = w.get("startTs", -1)
        en = w.get("endTs", 1e18)
        if st is not None and ts >= st and ts < (en if en is not None else 1e18):
            return w.get("rule", "normal")
    return "normal"


def _is_allowed(shape: str, rule: str) -> bool:
    if rule == "normal":
        return True
    if rule == "onlySquares":
        return shape == "square"
    if rule == "onlyCircles":
        return shape == "circle"
    if rule == "onlyTriangles":
        return shape == "triangle"
    return True


def _compute_server_summary(data: dict) -> dict:
    taps: list[dict] = data.get("taps", [])
    targets: list[dict] = data.get("targets", [])
    distractions: list[dict] = data.get("distractions", [])
    rule_windows: list[dict] = data.get("rule_windows") or []

    # Build eligible targets: non-fake AND allowed under rule at spawn
    eligible = []
    for t in targets:
        if t.get("fake"):
            continue
        rule = t.get("ruleAtSpawn", "normal")
        shape = t.get("type")
        if _is_allowed(shape, rule):
            eligible.append(t)

    # Build target index
    tgt_by_id = {t.get("id"): t for t in targets}

    # Determine first correct hit per eligible target under rule at tap time
    hit_by_target: dict[int, dict] = {}
    for tp in taps:
        tid = tp.get("targetId")
        if tid is None:
            continue
        tgt = tgt_by_id.get(tid)
        if not tgt or tgt.get("fake"):
            continue
        ts_tap = tp.get("ts", 0)
        rule_tap = tp.get("ruleAtTap", "normal")
        shape = tgt.get("type")
        if _is_allowed(shape, rule_tap) and tp.get("correct") and tid not in hit_by_target:
            hit_by_target[tid] = {"tapTs": ts_tap, "spawnTs": tgt.get("spawnTs", 0)}

    # Overall
    total_eligible = len(eligible)
    hits = len(hit_by_target)
    misses = total_eligible - hits
    accuracy = round((hits / total_eligible) * 100) if total_eligible > 0 else 0

    rts = [h["tapTs"] - h["spawnTs"] for h in hit_by_target.values()]
    avg_rt = round(sum(rts) / len(rts)) if rts else None

    # Impulsivity: premature + fake taps
    premature = sum(1 for t in taps if t.get("premature"))
    fake_taps = sum(1 for t in taps if t.get("targetId") is not None and tgt_by_id.get(t.get("targetId"), {}).get("fake"))
    impulsive = premature + fake_taps

    errors = sum(1 for t in taps if not t.get("correct"))

    # Helper to compute window stats
    WINDOW_MS = 5000
    def window_stats(start: float, end: float) -> dict:
        # eligible targets in window
        elig = [t for t in eligible if start <= t.get("spawnTs", 0) < end]
        denom = len(elig)
        # hits: of those, did they get a valid tap?
        hit_ids = {t.get("id") for t in elig if t.get("id") in hit_by_target and start <= hit_by_target[t.get("id")]["tapTs"] < end}
        h = len(hit_ids)
        acc = (h / denom) if denom > 0 else None
        hit_rts = [hit_by_target[tid]["tapTs"] - tgt_by_id[tid].get("spawnTs", 0) for tid in hit_ids]
        avg_rt_w = round(sum(hit_rts) / len(hit_rts)) if hit_rts else None
        return {"targets": denom, "hits": h, "acc": acc, "avgRt": avg_rt_w}

    # Per distraction
    per_distraction = []
    for d in distractions:
        st = d.get("startTs", 0)
        pre = window_stats(st - WINDOW_MS, st)
        post = window_stats(st, st + WINDOW_MS)
        per_distraction.append({
            "id": d.get("id"),
            "kind": d.get("kind"),
            "startTs": st,
            "duration": d.get("duration"),
            "pre": pre,
            "post": post,
        })

    return {
        "totalEligible": total_eligible,
        "hits": hits,
        "misses": misses,
        "errors": errors,
        "accuracy": accuracy,
        "avgRt": avg_rt,
        "impulsive": impulsive,
        "perDistraction": per_distraction,
    }


@app.post("/api/game/submit")
async def submit_game(payload: GameSubmission):
    record = payload.model_dump()
    record["type"] = "game"
    record["server_received"] = datetime.utcnow().isoformat() + "Z"
    # Compute authoritative server summary
    try:
        server_summary = _compute_server_summary(record)
        record["server_summary"] = server_summary
    except Exception as e:
        record["server_summary_error"] = str(e)
    try:
        with open(GAME_RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save game data: {e}")
    return {"saved": True, "server_summary": record.get("server_summary")}


# ===================================
# Video Analysis Endpoints
# ===================================

VIDEO_RESULTS_PATH = DATA_DIR / "video_results.jsonl"
VIDEO_DIR = DATA_DIR / "video"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


class VideoSubmission(BaseModel):
    """Video analysis submission with complete timeline and summary."""
    session_id: str
    trigger_video: str
    duration_seconds: float
    timeline: List[dict]  # Time-series data
    summary: dict  # Aggregated metrics


@app.get("/video")
async def serve_video_page():
    """Serve the video analysis page."""
    video_html_path = FRONTEND_DIR / "video.html"
    if not video_html_path.exists():
        raise HTTPException(status_code=404, detail="video.html not found")
    return FileResponse(video_html_path)


@app.post("/api/video/start-session")
async def start_video_session():
    """
    Start a new video analysis session.
    Resets the analyzer's state.
    """
    try:
        analyzer = get_video_analyzer()
        analyzer.start_analysis()
        return {"status": "started", "message": "Video analysis session started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")


@app.get("/api/video/trigger")
async def get_trigger_video():
    """Stream the trigger video file."""
    video_path = VIDEO_DIR / "hack.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Trigger video not found")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


class FrameProcessRequest(BaseModel):
    """Single frame processing request."""
    frame_base64: str
    frame_number: int
    session_id: Optional[str] = None


@app.post("/api/video/process-frame")
async def process_video_frame(request: FrameProcessRequest):
    """
    Process a single frame through all video analysis models.
    Returns real-time analysis results for emotion, blink, iris, and gaze.
    """
    try:
        # Decode base64 image
        img_data = base64.b64decode(request.frame_base64.split(',')[1] if ',' in request.frame_base64 else request.frame_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Get analyzer and process frame
        analyzer = get_video_analyzer()
        result = analyzer.process_frame(frame, request.frame_number)
        
        return result
        
    except Exception as e:
        print(f"Error processing frame {request.frame_number}: {e}")
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")


@app.post("/api/video/submit")
async def submit_video_analysis(submission: VideoSubmission):
    """
    Submit video analysis results with timeline and summary.
    Saves to video_results.jsonl for later processing.
    Also returns server-computed summary from VideoAnalyzer.
    """
    try:
        # Get server-side summary
        analyzer = get_video_analyzer()
        server_summary = analyzer.get_summary()
        
        record = {
            "session_id": submission.session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trigger_video": submission.trigger_video,
            "duration_seconds": submission.duration_seconds,
            "timeline": submission.timeline,
            "client_summary": submission.summary,  # From client calculation
            "server_summary": server_summary  # From VideoAnalyzer
        }
        
        with open(VIDEO_RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video data: {e}")
    
    return {
        "saved": True,
        "session_id": submission.session_id,
        "timeline_entries": len(submission.timeline),
        "client_summary": submission.summary,
        "server_summary": server_summary
    }


if __name__ == "__main__":
    # Optional: run with `python backend/main.py` for quick testing
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
