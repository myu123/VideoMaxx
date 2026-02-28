"""Job management routes: upload, process, status, results."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from backend.config import RAW_VIDEOS_DIR, OUTPUTS_DIR
from backend.pipeline.transcribe import transcribe, save_transcript, load_transcript
from backend.pipeline.candidates import generate_candidates, save_candidates, load_candidates
from backend.pipeline.features import extract_features, save_features, load_features
from backend.pipeline.ml import score_candidates, explain_candidate
from backend.pipeline.clips import select_top_clips, render_clips
from backend.pipeline.timeline import build_timeline, save_timeline, load_timeline

import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"])

# In-memory job status tracking
_job_status: dict[str, dict] = {}


def _update_status(job_id: str, **kwargs):
    """Update job status fields and record timestamp."""
    _job_status[job_id].update(kwargs)
    _job_status[job_id]["updated_at"] = time.time()
    if "log" in kwargs:
        logs = _job_status[job_id].setdefault("logs", [])
        logs.append({"time": time.time(), "message": kwargs["log"]})


def _get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _get_video_info(video_path: str) -> dict:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size:stream=width,height,r_frame_rate,codec_name",
        "-select_streams", "v:0",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        # Parse frame rate fraction
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = round(int(num) / max(int(den), 1), 1)
        else:
            fps = float(fps_str)

        return {
            "duration": float(fmt.get("duration", 0)),
            "size_mb": round(int(fmt.get("size", 0)) / 1048576, 1),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": fps,
            "codec": stream.get("codec_name", "unknown"),
        }
    except Exception:
        return {"duration": _get_video_duration(video_path)}


def _process_video(job_id: str, video_path: Path):
    """Full pipeline: transcribe -> candidates -> features -> score -> clips."""
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    try:
        # ── Probe video ────────────────────────────────────────
        _update_status(job_id, stage="probing", progress=2,
                       log="Analyzing video file...")
        video_info = _get_video_info(str(video_path))
        _update_status(job_id, video_info=video_info,
                       log=f"Video: {video_info.get('width', '?')}x{video_info.get('height', '?')} "
                           f"@ {video_info.get('fps', '?')}fps, "
                           f"{video_info.get('duration', 0):.0f}s, "
                           f"{video_info.get('size_mb', '?')}MB, "
                           f"codec={video_info.get('codec', '?')}")

        # ── Check CUDA ─────────────────────────────────────────
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                _update_status(job_id, log=f"GPU detected: {gpu_name} (CUDA {torch.version.cuda})")
            else:
                _update_status(job_id, log="CUDA not available — using CPU (slower)")
        except ImportError:
            _update_status(job_id, log="PyTorch not found — using CPU")

        # ── Step 1: Transcribe ─────────────────────────────────
        _update_status(job_id, stage="transcribing", progress=5,
                       log="Loading Whisper model...")
        t0 = time.time()
        segments = transcribe(video_path)
        transcribe_time = time.time() - t0
        save_transcript(segments, output_dir / "transcript.json")

        total_words = sum(len(s["text"].split()) for s in segments)
        _update_status(job_id, progress=25,
                       log=f"Transcription complete in {transcribe_time:.1f}s — "
                           f"{len(segments)} segments, {total_words} words")

        # ── Step 2: Generate candidates ────────────────────────
        _update_status(job_id, stage="generating_candidates", progress=28,
                       log="Generating candidate highlight windows...")
        candidates = generate_candidates(segments)
        save_candidates(candidates, output_dir / "candidates.csv")

        if not candidates:
            _update_status(job_id, stage="error",
                           error="No candidate segments generated — video may be too short")
            return

        avg_dur = np.mean([c["end"] - c["start"] for c in candidates])
        _update_status(job_id, progress=30,
                       candidates_count=len(candidates),
                       log=f"Generated {len(candidates)} candidate segments "
                           f"(avg {avg_dur:.0f}s each)")

        # ── Step 3: Extract features ───────────────────────────
        _update_status(job_id, stage="extracting_features", progress=32,
                       sub_stage="text_embeddings",
                       log="Computing text embeddings (sentence-transformers)...")

        # We need to hook into the feature extraction to provide granular updates
        # For now, the feature extraction runs as a batch
        t0 = time.time()
        embeddings, numerics = extract_features(candidates, str(video_path))
        features_time = time.time() - t0
        save_features(embeddings, numerics, output_dir)

        _update_status(job_id, progress=70,
                       log=f"Feature extraction complete in {features_time:.1f}s — "
                           f"{numerics.shape[1]} features per segment "
                           f"({embeddings.shape[1]}d embeddings + {numerics.shape[1]} numeric)")

        # ── Step 4: Score candidates ───────────────────────────
        _update_status(job_id, stage="scoring", progress=72,
                       log="Scoring candidates with ML model...")
        scores = score_candidates(embeddings, numerics)
        scores_data = []
        for i, c in enumerate(candidates):
            reasons = explain_candidate(
                embeddings[i:i+1],
                numerics[i:i+1],
                c["text"],
            )
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

        top_score = max(scores)
        avg_score = float(np.mean(scores))
        _update_status(job_id, progress=78,
                       log=f"Scoring complete — top score: {top_score:.1f}, "
                           f"avg: {avg_score:.1f}")

        # ── Step 5: Build timeline ─────────────────────────────
        _update_status(job_id, stage="building_timeline", progress=80,
                       log="Building engagement timeline...")
        duration = video_info.get("duration", _get_video_duration(str(video_path)))
        timeline = build_timeline(candidates, scores, duration)
        save_timeline(timeline, output_dir / "timeline.json")
        _update_status(job_id, log=f"Timeline built — {len(timeline)} data points")

        # ── Step 6: Select and render top clips ────────────────
        _update_status(job_id, stage="rendering_clips", progress=83,
                       log="Selecting top non-overlapping clips...")
        selected = select_top_clips(candidates, scores)
        _update_status(job_id,
                       log=f"Selected {len(selected)} clips, rendering with subtitles...")

        t0 = time.time()
        rendered = render_clips(video_path, selected, segments, output_dir)
        render_time = time.time() - t0
        _update_status(job_id, progress=95,
                       log=f"Rendered {len(rendered)} clips in {render_time:.1f}s")

        # ── Save final results ─────────────────────────────────
        results = {
            "job_id": job_id,
            "video_duration": duration,
            "total_candidates": len(candidates),
            "clips": [],
        }
        for clip in rendered:
            clip_result = {
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
            }
            results["clips"].append(clip_result)

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        total_time = time.time() - start_time
        _update_status(job_id, stage="complete", progress=100,
                       log=f"Pipeline complete in {total_time:.1f}s — "
                           f"{len(rendered)} highlight clips ready")
        logger.info("Job %s completed in %.1fs", job_id, total_time)

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        _update_status(job_id, stage="error", error=str(e),
                       log=f"ERROR: {str(e)}")


@router.post("/jobs/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video and start processing."""
    job_id = str(uuid.uuid4())[:8]
    video_path = RAW_VIDEOS_DIR / f"{job_id}_{file.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size_mb = round(video_path.stat().st_size / 1048576, 1)

    _job_status[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "video_path": str(video_path),  # internal only, filtered from API responses
        "stage": "queued",
        "progress": 0,
        "error": None,
        "video_info": None,
        "candidates_count": None,
        "logs": [{"time": time.time(), "message": f"Uploaded {file.filename} ({file_size_mb}MB)"}],
        "updated_at": time.time(),
    }

    # Run in a separate thread so the event loop stays free to serve status polls
    thread = threading.Thread(target=_process_video, args=(job_id, video_path), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return [{k: v for k, v in job.items() if k != "video_path"} for job in _job_status.values()]


@router.get("/jobs/{job_id}/status")
async def job_status(job_id: str):
    if job_id not in _job_status:
        raise HTTPException(404, "Job not found")
    # Return a copy without internal fields (video_path is a local filesystem path)
    data = {k: v for k, v in _job_status[job_id].items() if k != "video_path"}
    return data


@router.get("/jobs/{job_id}/results")
async def job_results(job_id: str):
    results_path = OUTPUTS_DIR / job_id / "results.json"
    if not results_path.exists():
        raise HTTPException(404, "Results not ready")
    with open(results_path) as f:
        return json.load(f)


@router.get("/jobs/{job_id}/timeline")
async def job_timeline(job_id: str):
    tl_path = OUTPUTS_DIR / job_id / "timeline.json"
    if not tl_path.exists():
        raise HTTPException(404, "Timeline not ready")
    with open(tl_path) as f:
        return json.load(f)


@router.get("/jobs/{job_id}/candidates")
async def job_candidates(job_id: str):
    cand_path = OUTPUTS_DIR / job_id / "candidates.csv"
    if not cand_path.exists():
        raise HTTPException(404, "Candidates not ready")
    candidates = load_candidates(cand_path)

    # Attach scores if available
    scores_path = OUTPUTS_DIR / job_id / "scores.json"
    if scores_path.exists():
        with open(scores_path) as f:
            scores_data = json.load(f)
        scores_map = {s["candidate_id"]: s for s in scores_data}
        for c in candidates:
            if c["candidate_id"] in scores_map:
                c["score"] = scores_map[c["candidate_id"]]["score"]
                c["reasons"] = scores_map[c["candidate_id"]]["reasons"]

    return candidates


@router.get("/jobs/{job_id}/transcript")
async def job_transcript(job_id: str):
    t_path = OUTPUTS_DIR / job_id / "transcript.json"
    if not t_path.exists():
        raise HTTPException(404, "Transcript not ready")
    return load_transcript(t_path)


@router.get("/jobs/{job_id}/clips/{clip_filename}")
async def download_clip(job_id: str, clip_filename: str):
    clip_path = OUTPUTS_DIR / job_id / "clips" / clip_filename
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(str(clip_path), media_type="video/mp4", filename=clip_filename)
