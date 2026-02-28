"""Candidate window generation from transcript segments."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from backend.config import CANDIDATE_MIN_DURATION, CANDIDATE_MAX_DURATION

logger = logging.getLogger(__name__)


def generate_candidates(
    transcript: List[Dict[str, Any]],
    min_duration: float = CANDIDATE_MIN_DURATION,
    max_duration: float = CANDIDATE_MAX_DURATION,
) -> List[Dict[str, Any]]:
    """Sliding-merge strategy over transcript segments.

    Returns:
        [{"candidate_id": int, "start": float, "end": float, "text": str}, ...]
    """
    if not transcript:
        return []

    candidates: List[Dict[str, Any]] = []
    cid = 0

    for i in range(len(transcript)):
        merged_text_parts: List[str] = []
        start = transcript[i]["start"]
        end = transcript[i]["end"]

        for j in range(i, len(transcript)):
            seg = transcript[j]
            end = seg["end"]
            merged_text_parts.append(seg["text"])
            duration = end - start

            if duration >= min_duration:
                if duration <= max_duration:
                    candidates.append({
                        "candidate_id": cid,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": " ".join(merged_text_parts),
                    })
                    cid += 1
                break  # stop extending this window once we pass min

    logger.info("Generated %d candidate windows", len(candidates))
    return candidates


def save_candidates(candidates: List[Dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidates[0].keys())
        writer.writeheader()
        writer.writerows(candidates)


def load_candidates(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["candidate_id"] = int(row["candidate_id"])
            row["start"] = float(row["start"])
            row["end"] = float(row["end"])
            rows.append(row)
        return rows
