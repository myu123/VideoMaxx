"""Feature extraction: text embeddings + audio statistics + video features."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import librosa
import soundfile as sf
from sentence_transformers import SentenceTransformer

from backend.config import (
    AUDIO_SAMPLE_RATE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)
from backend.pipeline.video_features import (
    extract_video_features_batch,
    VIDEO_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Loaded sentence-transformer: %s", EMBEDDING_MODEL_NAME)
    return _embed_model


# ── Text features ──────────────────────────────────────────────────

def _text_stats(text: str, duration: float) -> Dict[str, float]:
    words = text.split()
    word_count = len(words)
    return {
        "word_count": float(word_count),
        "speaking_rate": word_count / max(duration, 0.1),
        "question_count": float(text.count("?")),
        "exclamation_count": float(text.count("!")),
        "avg_word_length": float(np.mean([len(w) for w in words]) if words else 0.0),
    }


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return (N, 384) embedding matrix."""
    model = _get_embed_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


# ── Audio features ─────────────────────────────────────────────────

def _extract_audio_segment(
    video_path: str | Path,
    start: float,
    end: float,
) -> np.ndarray:
    """Extract audio segment from video via ffmpeg, return numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-vn",
        tmp_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    audio, _ = sf.read(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return audio.astype(np.float32)


def _audio_stats(audio: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> Dict[str, float]:
    if len(audio) < sr * 0.1:
        return {
            "rms_mean": 0.0,
            "rms_max": 0.0,
            "spectral_centroid_mean": 0.0,
            "zcr_mean": 0.0,
        }
    rms = librosa.feature.rms(y=audio)[0]
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    return {
        "rms_mean": float(np.mean(rms)),
        "rms_max": float(np.max(rms)),
        "spectral_centroid_mean": float(np.mean(sc)),
        "zcr_mean": float(np.mean(zcr)),
    }


# ── Combined feature names ────────────────────────────────────────

AUDIO_TEXT_FEATURE_NAMES = [
    "word_count",
    "speaking_rate",
    "question_count",
    "exclamation_count",
    "avg_word_length",
    "rms_mean",
    "rms_max",
    "spectral_centroid_mean",
    "zcr_mean",
]

# Full numeric feature list: audio/text + video
NUMERIC_FEATURE_NAMES = AUDIO_TEXT_FEATURE_NAMES + VIDEO_FEATURE_NAMES


def extract_features(
    candidates: List[Dict[str, Any]],
    video_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for a list of candidates.

    Returns:
        embeddings: (N, 384)  text embeddings
        numerics:   (N, 38)   numeric features (9 audio/text + 29 video)
    """
    texts = [c["text"] for c in candidates]
    embeddings = embed_texts(texts)

    # Audio + text features
    logger.info("Extracting audio/text features for %d candidates...", len(candidates))
    audio_text_rows = []
    for c in candidates:
        duration = c["end"] - c["start"]
        ts = _text_stats(c["text"], duration)

        try:
            audio = _extract_audio_segment(video_path, c["start"], c["end"])
            aus = _audio_stats(audio)
        except Exception as e:
            logger.warning("Audio extraction failed for candidate %s: %s", c.get("candidate_id"), e)
            aus = {k: 0.0 for k in ["rms_mean", "rms_max", "spectral_centroid_mean", "zcr_mean"]}

        row = [ts[k] for k in AUDIO_TEXT_FEATURE_NAMES[:5]] + \
              [aus[k] for k in AUDIO_TEXT_FEATURE_NAMES[5:]]
        audio_text_rows.append(row)

    audio_text_arr = np.array(audio_text_rows, dtype=np.float32)

    # Video features
    logger.info("Extracting video features for %d candidates...", len(candidates))
    video_arr = extract_video_features_batch(video_path, candidates)

    # Concatenate: [audio_text (9) | video (29)] = 38 numeric features
    numerics = np.hstack([audio_text_arr, video_arr])

    return embeddings, numerics


def save_features(
    embeddings: np.ndarray,
    numerics: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "numerics.npy", numerics)


def load_features(output_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    output_dir = Path(output_dir)
    embeddings = np.load(output_dir / "embeddings.npy")
    numerics = np.load(output_dir / "numerics.npy")
    return embeddings, numerics
