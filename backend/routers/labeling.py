"""Labeling routes for manual highlight annotation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import LABELS_CSV

logger = logging.getLogger(__name__)
router = APIRouter(tags=["labeling"])


class LabelRequest(BaseModel):
    video_id: str
    start: float
    end: float
    label: int  # 1 = highlight, 0 = not highlight


@router.post("/labels")
async def save_label(req: LabelRequest):
    """Save a label for a candidate segment."""
    if req.label not in (0, 1):
        raise HTTPException(400, "Label must be 0 or 1")

    file_exists = LABELS_CSV.exists()
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_id", "start", "end", "label"])
        writer.writerow([req.video_id, req.start, req.end, req.label])

    return {"status": "saved"}


@router.get("/labels")
async def get_labels(video_id: Optional[str] = None):
    """Get all labels, optionally filtered by video_id."""
    if not LABELS_CSV.exists():
        return []

    labels = []
    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if video_id and row["video_id"] != video_id:
                continue
            labels.append({
                "video_id": row["video_id"],
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": int(row["label"]),
            })

    return labels


@router.get("/labels/stats")
async def label_stats():
    """Get labeling statistics."""
    if not LABELS_CSV.exists():
        return {"total": 0, "highlights": 0, "non_highlights": 0}

    total = highlights = 0
    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if int(row["label"]) == 1:
                highlights += 1

    return {
        "total": total,
        "highlights": highlights,
        "non_highlights": total - highlights,
    }
