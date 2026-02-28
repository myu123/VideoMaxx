"""Clip selection, caption generation, and rendering via ffmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from backend.config import MAX_CLIPS, OVERLAP_IOU_THRESHOLD, MIN_START_DISTANCE

logger = logging.getLogger(__name__)


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0.0, inter_end - inter_start)
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / max(union, 1e-6)


def select_top_clips(
    candidates: List[Dict[str, Any]],
    scores: list | Any,
    max_clips: int = MAX_CLIPS,
) -> List[Dict[str, Any]]:
    """Select top non-overlapping clips by score."""
    scored = []
    for i, c in enumerate(candidates):
        scored.append({**c, "score": float(scores[i])})

    scored.sort(key=lambda x: x["score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    for c in scored:
        overlap = False
        for s in selected:
            if _iou(c["start"], c["end"], s["start"], s["end"]) > OVERLAP_IOU_THRESHOLD:
                overlap = True
                break
            if abs(c["start"] - s["start"]) < MIN_START_DISTANCE:
                overlap = True
                break
        if not overlap:
            selected.append(c)
        if len(selected) >= max_clips:
            break

    return selected


def generate_srt(
    transcript: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> str:
    """Generate SRT subtitle content for a clip window."""
    srt_parts = []
    idx = 1
    for seg in transcript:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue
        s = max(seg["start"] - clip_start, 0)
        e = seg["end"] - clip_start
        srt_parts.append(
            f"{idx}\n"
            f"{_format_srt_time(s)} --> {_format_srt_time(e)}\n"
            f"{seg['text']}\n"
        )
        idx += 1
    return "\n".join(srt_parts)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_clip(
    video_path: str | Path,
    clip: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    output_dir: str | Path,
) -> Dict[str, str]:
    """Render a single clip with burned subtitles.

    Returns dict with paths to clip and srt files.
    """
    output_dir = Path(output_dir)
    clips_dir = output_dir / "clips"
    captions_dir = output_dir / "captions"
    clips_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    cid = clip["candidate_id"]
    srt_content = generate_srt(transcript, clip["start"], clip["end"])
    srt_path = captions_dir / f"clip_{cid}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    raw_clip_path = clips_dir / f"clip_{cid}_raw.mp4"
    final_clip_path = clips_dir / f"clip_{cid}.mp4"

    # Cut clip
    cmd_cut = [
        "ffmpeg", "-y",
        "-ss", str(clip["start"]),
        "-to", str(clip["end"]),
        "-i", str(video_path),
        "-c", "copy",
        str(raw_clip_path),
    ]
    subprocess.run(cmd_cut, capture_output=True, check=True)

    # Burn subtitles (need to escape path for ffmpeg filter)
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    cmd_subs = [
        "ffmpeg", "-y",
        "-i", str(raw_clip_path),
        "-vf", f"subtitles='{srt_escaped}'",
        str(final_clip_path),
    ]
    result = subprocess.run(cmd_subs, capture_output=True)
    if result.returncode != 0:
        logger.warning("Subtitle burn failed, copying raw clip. stderr: %s", result.stderr.decode()[-500:])
        # Fallback: just use the raw clip
        import shutil
        shutil.copy2(raw_clip_path, final_clip_path)

    # Clean up raw
    raw_clip_path.unlink(missing_ok=True)

    return {
        "clip_path": str(final_clip_path),
        "srt_path": str(srt_path),
    }


def render_clips(
    video_path: str | Path,
    selected_clips: List[Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    output_dir: str | Path,
) -> List[Dict[str, Any]]:
    """Render all selected clips. Returns enriched clip list."""
    results = []
    for clip in selected_clips:
        paths = render_clip(video_path, clip, transcript, output_dir)
        results.append({**clip, **paths})
        logger.info("Rendered clip %s (%.1fs - %.1fs)", clip["candidate_id"], clip["start"], clip["end"])
    return results
