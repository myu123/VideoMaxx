"""Engagement timeline generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from backend.config import TIMELINE_WINDOW, TIMELINE_STEP

logger = logging.getLogger(__name__)


def build_timeline(
    candidates: List[Dict[str, Any]],
    scores: np.ndarray | list,
    video_duration: float,
) -> List[Dict[str, float]]:
    """Build engagement timeline by interpolating candidate scores over time.

    Returns:
        [{"time": t, "score": s}, ...]
    """
    scores = np.asarray(scores, dtype=np.float64)
    timeline = []

    t = 0.0
    while t < video_duration:
        window_end = t + TIMELINE_WINDOW
        overlapping_scores = []

        for i, c in enumerate(candidates):
            c_start = float(c["start"])
            c_end = float(c["end"])
            # Check overlap
            if c_start < window_end and c_end > t:
                overlapping_scores.append(scores[i])

        if overlapping_scores:
            point_score = float(np.mean(overlapping_scores))
        else:
            point_score = 0.0

        timeline.append({
            "time": round(t, 1),
            "score": round(point_score, 2),
        })
        t += TIMELINE_STEP

    return timeline


def save_timeline(timeline: List[Dict[str, float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(timeline, f, indent=2)


def load_timeline(path: str | Path) -> List[Dict[str, float]]:
    with open(path, "r") as f:
        return json.load(f)
