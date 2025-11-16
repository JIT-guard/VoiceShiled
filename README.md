# VoiceShield

VoiceShield is a small FastAPI web application that accepts short audio or video clips, runs the audio through OpenAI's Whisper for transcription, classifies every speech segment with a hate-speech detector, and replaces toxic portions of the soundtrack with a censor beep. The processed audio is muxed back into the original video (or provided as a clean `.wav` file for audio uploads) so you can safely share the result.

## Features

- 🎙️ Upload common audio/video formats straight from the browser.
- 🧠 Automatic transcription with locally hosted Whisper (`tiny`, `base`, `small`, or `medium`).
- ⚠️ Toxic-language detection using the Hugging Face `unitary/toxic-bert` classifier (configurable threshold).
- 🎧 Friendly chime overlay on every flagged segment plus a transcript preview highlighting what was censored.
- ▶️ Inline playback of the censored audio/video before you download it.
- 🌐 Single FastAPI service that also serves the static frontend; deployable to Render or any container host.

## Requirements

- Python 3.10+
- `ffmpeg` binaries available on your `PATH` (required by MoviePy and Pydub to read/write media).
- Sufficient CPU/RAM for the Whisper model you choose (Render's `medium` instance or better is recommended).

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start the development server:

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser, drop an audio/video file, pick the Whisper size and toxicity threshold, and wait for the censored download link. For the first request the models will download, so expect a longer cold start.

### API overview

- `POST /api/process` – multipart form upload with fields:
  - `file`: media file.
  - `whisper_model`: `tiny | base | small | medium`.
  - `threshold`: detection probability between `0.1` and `0.95`.
- `GET /api/status/{job_id}` – returns the stored metadata and transcript for a processed job.
- `GET /api/download/{job_id}` – downloads the censored media.

### Deploying to Render

1. Push this repo to GitHub/GitLab.
2. Create a new **Web Service** in Render that points to the repository.
3. Set **Runtime** to Python 3.10+ and **Build Command** to `pip install -r requirements.txt`.
4. Set **Start Command** to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. Add a Render Secret for `PYTHONUNBUFFERED=1` (optional) and make sure an instance type with enough RAM is selected (Whisper `medium` needs at least 8 GB).
6. Whisper/HF models download on first run, so keep the service warm for better UX.

If you wrap the service in Docker, ensure the image installs `ffmpeg` (e.g. `apt-get install ffmpeg`) before running the app.

## Known limitations

- Whisper and the HF classifier run entirely on CPU by default; processing long clips will take time.
- Currently, every flagged segment is fully replaced by a beep. Word-level muting would require Whisper word timestamps and a finer-grained alignment step.
- Uploaded and processed files persist on disk until manually rotated; consider adding a cron job or background worker for cleanup in production.
