# ML_Xai_major_project — Stage 1: PHQ-8 Questionnaire

This minimal FastAPI + HTML/CSS/JS app implements the first stage of your plan: a PHQ-8 questionnaire. It computes the score client-side, persists it server-side, and shows severity. Results are appended to `backend/data/phq8_results.jsonl` for later stages (game metrics, video analysis, GenAI report).

## What’s here

- Backend: FastAPI app in `backend/main.py`, serves the frontend and provides `POST /api/phq8/submit`.
- Frontend: Static HTML/CSS/JS in `frontend/`.
- Data: Results stored as JSONL in `backend/data/phq8_results.jsonl`.

## Run locally (Windows PowerShell)

```powershell
# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Option 1: use uvicorn directly
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Option 2: run the file (also uses uvicorn)
python backend\main.py
```

Open http://127.0.0.1:8000 in your browser. The API is at http://127.0.0.1:8000/api/.

## API

- `POST /api/phq8/submit`
  - Body: `{ "session_id": string, "answers": number[8] }` (each 0–3)
  - Response: `{ "score": number, "severity": string, "saved": boolean }`

Severity bands: 0–4 None, 5–9 Mild, 10–14 Moderate, 15–19 Moderately Severe, 20–24 Severe.

## Notes

- CORS is currently permissive. Since the frontend is served by the same app, this is okay for dev; tighten for prod.
- If backend persistence fails, the client still shows your local score. The API response will include `saved: false`.
- Next stages (mini-game, video processing, GenAI summarization) can use the stored `session_id` to correlate data across steps.
