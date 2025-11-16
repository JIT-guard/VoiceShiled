from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .processing import MediaProcessor

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="VoiceShield", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = MediaProcessor(BASE_DIR)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/process")
async def process_media(
    file: UploadFile = File(...),
    threshold: float = Form(0.6),
    whisper_model: str = Form("base"),
) -> dict:
    try:
        result = await processor.process_upload(file, threshold, whisper_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job_id = result["job_id"]
    return {
        "jobId": job_id,
        "downloadUrl": f"/api/download/{job_id}",
        "metadata": result["metadata"],
    }


@app.get("/api/status/{job_id}")
async def job_status(job_id: str) -> dict:
    metadata = _load_metadata(job_id)
    return {"jobId": job_id, "metadata": metadata}


@app.get("/api/download/{job_id}")
async def download_result(job_id: str) -> FileResponse:
    metadata = _load_metadata(job_id)
    filename = metadata.get("download_filename", f"{job_id}.wav")
    output_path = processor.processed_dir / filename
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed file missing")
    return FileResponse(path=output_path, filename=filename)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _load_metadata(job_id: str) -> dict:
    metadata_path = processor.processed_dir / f"{job_id}.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(metadata_path.read_text())
