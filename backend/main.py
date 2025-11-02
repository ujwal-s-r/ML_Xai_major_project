from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from processors.video_analyzer import VideoAnalyzer

# Note: Video processing has been temporarily removed/reworked. No processors are imported.

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "backend" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = DATA_DIR / "video"
TEMP_DIR = DATA_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = DATA_DIR / "phq8_results.jsonl"
VIDEO_RESULTS_PATH = DATA_DIR / "video_results.jsonl"
VIDEO_STATUS: dict[str, dict] = {}

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

# Video analyzer removed


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


def append_video_result(record: dict) -> None:
    with open(VIDEO_RESULTS_PATH, "a", encoding="utf-8") as f:
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


@app.get("/video")
async def video_page() -> FileResponse:
    """Serve the Video Analysis page."""
    video_page_path = FRONTEND_DIR / "video.html"
    if not video_page_path.exists():
        raise HTTPException(status_code=500, detail="Video page not found")
    return FileResponse(str(video_page_path))


@app.get("/api/video/trigger")
async def get_trigger_video() -> FileResponse:
    """Stream the trigger video file to the client."""
    video_path = VIDEO_DIR / "hack.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Trigger video not found")
    return FileResponse(str(video_path), media_type="video/mp4")


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


def _safe_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".")).strip()


@app.post("/api/video/upload")
async def upload_webcam_video(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Accept a recorded webcam video, save to temp, and extract frames.

    Returns JSON with frame count, fps, duration, and relative paths.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    session_dir_name = f"{_safe_filename(session_id)}_{ts}"
    out_dir = TEMP_DIR / session_dir_name
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    orig_name = (file.filename or "webcam.webm")
    ext = orig_name.split(".")[-1].lower() if "." in orig_name else "webm"
    if ext not in {"webm", "mp4", "mkv", "avi", "mov"}:
        ext = "webm"
    video_path = out_dir / f"webcam_recording.{ext}"

    # Save upload to disk in chunks
    try:
        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await file.close()

    # Extract frames with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Uploaded video could not be opened")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    count = 0
    ok, frame = cap.read()
    while ok:
        count += 1
        fname = frames_dir / f"frame_{count:06d}.jpg"
        cv2.imwrite(str(fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ok, frame = cap.read()
    cap.release()

    duration = (reported_total / fps) if fps and reported_total else None

    return JSONResponse(
        {
            "session_id": session_id,
            "saved_video": str(video_path.relative_to(BASE_DIR)),
            "frames_dir": str(frames_dir.relative_to(BASE_DIR)),
            "total_frames": count or reported_total,
            "reported_total_frames": reported_total,
            "fps": fps,
            "duration_seconds": duration,
            "message": "Upload received and frames extracted",
        }
    )


def _update_status(session_id: str, stage: str, processed: int, total: int, state: str = "running"):
    VIDEO_STATUS[session_id] = {
        "stage": stage,
        "processed": processed,
        "total": total,
        "state": state,
        "time": datetime.utcnow().isoformat() + "Z",
    }


def _run_processing(session_id: str, frames_dir: Path):
    analyzer = VideoAnalyzer()

    def on_progress(progress):
        _update_status(session_id, progress.stage, progress.processed, progress.total)

    try:
        # Blink stage
        _update_status(session_id, "blink", 0, 0, state="running")
        blink_res = analyzer.process_frames_blink(frames_dir, on_progress=on_progress)
        total_blink = blink_res["summary"].get("frames_processed", 0)
        _update_status(session_id, "blink", total_blink, total_blink, state="done")

        # Gaze stage
        _update_status(session_id, "gaze", 0, 0, state="running")
        gaze_res = analyzer.process_frames_gaze(frames_dir, on_progress=on_progress)
        total_gaze = gaze_res["summary"].get("frames_processed", 0)
        _update_status(session_id, "gaze", total_gaze, total_gaze, state="done")

        # Pupil stage
        _update_status(session_id, "pupil", 0, 0, state="running")
        pupil_res = analyzer.process_frames_pupil(frames_dir, on_progress=on_progress)
        total_pupil = pupil_res["summary"].get("frames_processed", 0)
        _update_status(session_id, "pupil", total_pupil, total_pupil, state="done")

        # Build combined summary record
        combined = {
            "type": "video",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "blink": blink_res.get("summary", {}),
            "gaze": gaze_res.get("summary", {}),
            "pupil": pupil_res.get("summary", {}),
            "version": 1,
        }

        # Print combined JSON summary to terminal for review
        try:
            print(json.dumps(combined, ensure_ascii=False))
        except Exception:
            # Printing failures should not break processing
            pass

        # Persist combined summary to video_results.jsonl for later dashboarding
        try:
            append_video_result(combined)
        except Exception as e:
            # Do not fail processing if persistence fails; log in server stdout
            print(f"Failed to persist video summary: {e}")

        # All done
        VIDEO_STATUS[session_id] = {
            "stage": "done",
            "processed": total_pupil,
            "total": total_pupil,
            "state": "done",
            "time": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        VIDEO_STATUS[session_id] = {
            "stage": VIDEO_STATUS.get(session_id, {}).get("stage", "blink"),
            "processed": 0,
            "total": 0,
            "state": "error",
            "error": str(e),
            "time": datetime.utcnow().isoformat() + "Z",
        }


@app.post("/api/video/process")
async def start_video_processing(background_tasks: BackgroundTasks, session_id: str = Form(...), frames_dir: str = Form(...)):
    # Resolve frames_dir safely
    frames_path = (BASE_DIR / frames_dir).resolve()
    if not str(frames_path).startswith(str(TEMP_DIR.resolve())):
        raise HTTPException(status_code=400, detail="frames_dir must be under temp directory")
    if not frames_path.exists():
        raise HTTPException(status_code=404, detail="frames_dir not found")

    # Start background task
    background_tasks.add_task(_run_processing, session_id, frames_path)
    _update_status(session_id, "blink", 0, 0, state="running")
    return {"started": True}


@app.get("/api/video/status/{session_id}")
async def get_video_status(session_id: str):
    st = VIDEO_STATUS.get(session_id)
    if not st:
        raise HTTPException(status_code=404, detail="No status for session")
    return st


# ===================================
# Dashboard & Results Endpoints
# ===================================

@app.get("/dashboard")
async def serve_dashboard_page():
    """Serve the dashboard visualization page."""
    dashboard_path = FRONTEND_DIR / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(dashboard_path)


@app.get("/api/results/{session_id}")
async def get_session_results(session_id: str):
    """
    Retrieve all results for a given session ID.
    Returns PHQ-8 and game data combined.
    """
    results = {
        "session_id": session_id,
        "phq8": None,
        "game": None
    }
    
    # Read PHQ-8 results
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    if record.get("session_id") == session_id:
                        results["phq8"] = record
                        break
        except Exception as e:
            print(f"Error reading PHQ-8 results: {e}")
    
    # Read game results
    if GAME_RESULTS_PATH.exists():
        try:
            with open(GAME_RESULTS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    if record.get("session_id") == session_id:
                        results["game"] = record
                        break
        except Exception as e:
            print(f"Error reading game results: {e}")
    
    # Check if any data was found
    if not any([results["phq8"], results["game"]]):
        raise HTTPException(status_code=404, detail=f"No data found for session {session_id}")
    
    return results


if __name__ == "__main__":
    # Optional: run with `python backend/main.py` for quick testing
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
