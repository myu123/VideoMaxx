"""Transcription module using faster-whisper."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from faster_whisper import WhisperModel

from backend.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        try:
            _model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info("Whisper model loaded on %s", WHISPER_DEVICE)
        except Exception:
            logger.warning("CUDA unavailable, falling back to CPU")
            _model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
    return _model


def transcribe(video_path: str | Path) -> List[Dict[str, Any]]:
    """Transcribe a video file and return list of segments.

    Returns:
        [{"start": float, "end": float, "text": str}, ...]
    """
    model = _get_model()
    segments_iter, info = model.transcribe(
        str(video_path),
        beam_size=5,
        word_timestamps=False,
        vad_filter=True,
    )
    logger.info("Detected language: %s (prob %.2f)", info.language, info.language_probability)

    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    return segments


def save_transcript(segments: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Persist transcript to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)


def load_transcript(path: str | Path) -> List[Dict[str, Any]]:
    """Load transcript from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
