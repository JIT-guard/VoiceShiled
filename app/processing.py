from __future__ import annotations

import asyncio
import json
import logging
import math
import mimetypes
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import whisper
from fastapi import UploadFile
from moviepy.editor import AudioFileClip, VideoFileClip
from pydub import AudioSegment
from pydub.generators import Sine
from transformers import pipeline

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


@dataclass
class SegmentResult:
    start: float
    end: float
    text: str
    score: float
    censored: bool


class WhisperModelCache:
    """Lazily loads and reuses Whisper models to avoid repeated cold starts."""

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models: Dict[str, whisper.Whisper] = {}
        self._lock = threading.Lock()

    def get(self, size: str) -> whisper.Whisper:
        allowed = {"tiny", "base", "small", "medium", "large-v2"}
        if size not in allowed:
            raise ValueError(f"Unsupported Whisper model '{size}'. Allowed: {', '.join(sorted(allowed))}")
        with self._lock:
            if size not in self._models:
                logger.info("Loading Whisper model '%s' on %s", size, self._device)
                self._models[size] = whisper.load_model(size, device=self._device)
        return self._models[size]


class ToxicityDetector:
    """Wraps a Hugging Face pipeline and normalises its output to a single score."""

    def __init__(self, model_name: str = "unitary/toxic-bert") -> None:
        self._pipeline = pipeline("text-classification", model=model_name, top_k=None)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, text: str) -> float:
        cleaned = text.strip()
        if not cleaned:
            return 0.0

        outputs = self._pipeline(cleaned[:512])
        # Some pipelines return List[Dict], others List[List[Dict]]
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]
        label_scores = {item["label"].lower(): item["score"] for item in outputs}

        candidate_labels = [label for label in label_scores if any(key in label for key in ("hate", "toxic", "abuse"))]
        if "label_1" in label_scores:
            candidate_labels.append("label_1")

        if not candidate_labels:
            # Fall back to pipeline's top prediction.
            return outputs[0]["score"] if outputs else 0.0

        return max(label_scores[label] for label in candidate_labels if label in label_scores)


class MediaProcessor:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.upload_dir = self.base_dir / "storage" / "uploads"
        self.processed_dir = self.base_dir / "storage" / "processed"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.models = WhisperModelCache()
        self.detector = ToxicityDetector()

    async def process_upload(
        self,
        upload: UploadFile,
        threshold: float,
        whisper_model: str,
    ) -> Dict[str, object]:
        job_id = uuid.uuid4().hex
        suffix = Path(upload.filename or "").suffix.lower()
        if not suffix:
            raise ValueError("Uploaded file must include an extension.")
        if suffix not in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
            raise ValueError("Only common audio/video formats are supported.")

        dest_path = self.upload_dir / f"{job_id}{suffix}"
        await self._save_upload(upload, dest_path)

        try:
            result = await asyncio.to_thread(
                self._process_file,
                job_id,
                dest_path,
                threshold,
                whisper_model,
            )
        finally:
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)

        return result

    async def _save_upload(self, upload: UploadFile, dest_path: Path) -> None:
        with dest_path.open("wb") as buffer:
            while chunk := await upload.read(1024 * 1024):
                buffer.write(chunk)
        await upload.close()

    def _process_file(
        self,
        job_id: str,
        input_path: Path,
        threshold: float,
        whisper_model: str,
    ) -> Dict[str, object]:
        input_kind = "video" if input_path.suffix.lower() in VIDEO_EXTENSIONS else "audio"

        audio_path = (
            self._extract_audio(input_path, job_id)
            if input_kind == "video"
            else self._ensure_wav(input_path, job_id)
        )

        try:
            transcript = self._transcribe(audio_path, whisper_model)
            expanded_segments = self._expand_segments(transcript["segments"])
            segments = self._detect_segments(expanded_segments, threshold)
            flagged_ranges = [
                (int(seg.start * 1000), int(seg.end * 1000))
                for seg in segments
                if seg.censored
            ]

            censored_audio_path = self._apply_censor(audio_path, flagged_ranges, job_id)

            if input_kind == "video":
                final_path = self.processed_dir / f"{job_id}.mp4"
                self._mux_with_video(input_path, censored_audio_path, final_path)
                censored_audio_path.unlink(missing_ok=True)
            else:
                final_path = censored_audio_path
            mime_type, _ = mimetypes.guess_type(final_path.name)

            metadata = {
                "job_id": job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input_kind": input_kind,
                "whisper_model": whisper_model,
                "toxicity_model": self.detector.model_name,
                "threshold": threshold,
                "segments": [asdict(seg) for seg in segments],
                "censored_segments": sum(1 for seg in segments if seg.censored),
                "download_filename": final_path.name,
                "output_extension": final_path.suffix,
                "output_mime": mime_type or "application/octet-stream",
            }
            metadata_path = self.processed_dir / f"{job_id}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

            return {"job_id": job_id, "metadata": metadata, "output_path": final_path}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Processing failed for job %s", job_id)
            raise RuntimeError("Failed to process the uploaded media.") from exc
        finally:
            if audio_path.exists() and audio_path != input_path:
                audio_path.unlink(missing_ok=True)

    def _extract_audio(self, video_path: Path, job_id: str) -> Path:
        audio_path = self.upload_dir / f"{job_id}_audio.wav"
        clip = VideoFileClip(str(video_path))
        try:
            clip.audio.write_audiofile(
                str(audio_path),
                fps=16000,
                codec="pcm_s16le",
                logger=None,
            )
        finally:
            clip.close()
        return audio_path

    def _ensure_wav(self, audio_path: Path, job_id: str) -> Path:
        if audio_path.suffix.lower() == ".wav":
            return audio_path

        wav_path = self.upload_dir / f"{job_id}_audio.wav"
        segment = AudioSegment.from_file(audio_path)
        segment.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")
        return wav_path

    def _transcribe(self, audio_path: Path, model_size: str) -> Dict[str, object]:
        model = self.models.get(model_size)
        logger.info("Transcribing %s with Whisper %s", audio_path.name, model_size)
        return model.transcribe(str(audio_path), word_timestamps=False, fp16=False)

    def _detect_segments(self, segments: Sequence[Dict[str, object]], threshold: float) -> List[SegmentResult]:
        results: List[SegmentResult] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            score = self.detector.score(text)
            results.append(
                SegmentResult(
                    start=start,
                    end=end,
                    text=text,
                    score=score,
                    censored=score >= threshold,
                )
            )
        return results

    def _expand_segments(self, segments: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        """Split Whisper segments into sentence-sized chunks with interpolated timings."""
        expanded: List[Dict[str, object]] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            sentences = self._split_sentences(text)
            if not sentences:
                continue
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            total_duration = max(0.05, end - start)
            weights = [max(1, len(s.strip())) for s in sentences]
            weight_sum = sum(weights) or len(sentences)
            cursor = start
            segment_start_index = len(expanded)
            for idx, (sentence, weight) in enumerate(zip(sentences, weights)):
                portion = weight / weight_sum
                duration = portion * total_duration
                sentence_end = min(end, cursor + duration)
                expanded.append({"text": sentence.strip(), "start": cursor, "end": sentence_end})
                cursor = sentence_end
            if len(expanded) > segment_start_index:
                expanded[-1]["end"] = end
        return expanded

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        cleaned = text.replace("\n", " ").strip()
        if not cleaned:
            return []
        sentence_re = re.compile(r"[^.!?]+[.!?…]*")
        matches = [match.strip() for match in sentence_re.findall(cleaned)]
        return [match for match in matches if match]

    def _apply_censor(
        self,
        audio_path: Path,
        flagged_ranges: Iterable[tuple[int, int]],
        job_id: str,
    ) -> Path:
        audio = AudioSegment.from_file(audio_path)
        for start_ms, end_ms in sorted(flagged_ranges):
            duration = max(50, end_ms - start_ms)
            beep = self._make_censor_tone(duration)
            # Replace the offending section with the censor tone.
            audio = audio[:start_ms] + beep + audio[end_ms:]

        output_path = self.processed_dir / f"{job_id}.wav"
        audio.export(output_path, format="wav")
        return output_path

    def _mux_with_video(self, video_path: Path, audio_path: Path, output_path: Path) -> None:
        video_clip = VideoFileClip(str(video_path))
        audio_clip = AudioFileClip(str(audio_path))
        try:
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=str(self.upload_dir / f"{output_path.stem}_temp-audio.m4a"),
                remove_temp=True,
                logger=None,
            )
        finally:
            video_clip.close()
            audio_clip.close()

    def _make_censor_tone(self, duration_ms: int) -> AudioSegment:
        """Approximate a live-TV 'uh-oh' sting using layered sine motifs."""
        duration_ms = max(900, duration_ms)

        # Low synth bed to keep things musical.
        bed = AudioSegment.silent(duration=duration_ms)
        for freq in (196, 247, 311):
            pad = Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-28)
            bed = bed.overlay(pad)

        # Descending motif reminiscent of broadcast whooshes.
        motif_notes = (988, 830, 659, 523)
        motif = AudioSegment.silent(duration=0)
        for freq in motif_notes:
            note = Sine(freq).to_audio_segment(duration=260).apply_gain(-12)
            motif += note.fade_in(30).fade_out(80)

        loops = max(1, math.ceil(duration_ms / max(1, len(motif))))
        sequence = AudioSegment.silent(duration=0)
        for _ in range(loops):
            sequence += motif
        sequence = sequence[:duration_ms]

        # Add a subtle sparkle over the top.
        sparkle = Sine(1400).to_audio_segment(duration=180).apply_gain(-30)
        sparkle_line = AudioSegment.silent(duration=0)
        while len(sparkle_line) < duration_ms:
            sparkle_line += sparkle.fade_in(10).fade_out(80)
        sparkle_line = sparkle_line[:duration_ms]

        sting = bed.overlay(sequence).overlay(sparkle_line).apply_gain(-3)
        fade = max(120, duration_ms // 4)
        sting = sting.fade_in(fade).fade_out(fade)
        return AudioSegment.silent(duration=duration_ms).overlay(sting)
