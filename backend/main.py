from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

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


if __name__ == "__main__":
    # Optional: run with `python backend/main.py` for quick testing
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
