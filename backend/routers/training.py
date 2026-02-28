"""Training and re-scoring routes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import OUTPUTS_DIR, ML_ARTIFACTS_DIR
from backend.pipeline.ml import train_model, score_candidates, explain_candidate
from backend.pipeline.features import load_features
from backend.pipeline.candidates import load_candidates
from backend.pipeline.clips import select_top_clips, render_clips
from backend.pipeline.timeline import build_timeline, save_timeline
from backend.pipeline.transcribe import load_transcript

import numpy as np
import subprocess

logger = logging.getLogger(__name__)
router = APIRouter(tags=["training"])


def _get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


@router.post("/train")
async def train():
    """Train model from all labeled data."""
    # Find all job dirs that have features
    job_dirs = []
    for d in OUTPUTS_DIR.iterdir():
        if d.is_dir() and (d / "embeddings.npy").exists():
            job_dirs.append(d)

    if not job_dirs:
        raise HTTPException(400, "No processed jobs with features found")

    try:
        metrics = train_model(job_dirs)
        return {"status": "trained", "metrics": metrics}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/jobs/{job_id}/rescore")
async def rescore_job(job_id: str):
    """Re-score a job using the trained model and re-render clips."""
    output_dir = OUTPUTS_DIR / job_id
    if not output_dir.exists():
        raise HTTPException(404, "Job not found")

    emb_path = output_dir / "embeddings.npy"
    num_path = output_dir / "numerics.npy"
    if not emb_path.exists():
        raise HTTPException(400, "Features not extracted for this job")

    embeddings, numerics = load_features(output_dir)
    candidates = load_candidates(output_dir / "candidates.csv")
    transcript = load_transcript(output_dir / "transcript.json")

    scores = score_candidates(embeddings, numerics)

    # Save updated scores
    scores_data = []
    for i, c in enumerate(candidates):
        reasons = explain_candidate(embeddings[i:i+1], numerics[i:i+1], c["text"])
        scores_data.append({
            "candidate_id": c["candidate_id"],
            "start": c["start"],
            "end": c["end"],
            "score": float(scores[i]),
            "text": c["text"],
            "reasons": reasons,
        })
    with open(output_dir / "scores.json", "w") as f:
        json.dump(scores_data, f, indent=2)

    # Find video path from job status or scan raw_videos
    from backend.routers.jobs import _job_status
    video_path = _job_status.get(job_id, {}).get("video_path")
    if not video_path:
        raise HTTPException(400, "Video path not found — upload may have been in a previous session")

    # Rebuild timeline
    duration = _get_video_duration(video_path)
    timeline = build_timeline(candidates, scores, duration)
    save_timeline(timeline, output_dir / "timeline.json")

    # Re-render clips
    selected = select_top_clips(candidates, scores)
    rendered = render_clips(video_path, selected, transcript, output_dir)

    results = {
        "job_id": job_id,
        "video_duration": duration,
        "total_candidates": len(candidates),
        "clips": [],
    }
    for clip in rendered:
        results["clips"].append({
            "candidate_id": clip["candidate_id"],
            "start": clip["start"],
            "end": clip["end"],
            "score": clip["score"],
            "text": clip["text"][:200],
            "reasons": next(
                (s["reasons"] for s in scores_data if s["candidate_id"] == clip["candidate_id"]),
                [],
            ),
            "clip_url": f"/outputs/{job_id}/clips/clip_{clip['candidate_id']}.mp4",
            "srt_url": f"/outputs/{job_id}/captions/clip_{clip['candidate_id']}.srt",
        })

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


@router.get("/model/status")
async def model_status():
    """Check if a trained model exists and return its metrics."""
    model_path = ML_ARTIFACTS_DIR / "model.joblib"
    metrics_path = ML_ARTIFACTS_DIR / "metrics.json"

    if not model_path.exists():
        return {"trained": False}

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {"trained": True, "metrics": metrics}
