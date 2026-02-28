"""Video-level feature extraction: motion, scene changes, faces, brightness,
colorfulness, visual complexity, and camera motion.

All features are extracted per candidate segment using sampled frames at low
resolution (320x240) and low FPS (2-3 fps) for speed.  A 60-second segment
processes in ~3-5 seconds on CPU.
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from backend.config import AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
_SAMPLE_FPS = 3          # frames per second to sample
_SAMPLE_W = 320
_SAMPLE_H = 240
_FACE_CONFIDENCE = 0.5   # DNN face detector threshold

# Lazy-loaded face detector (Haar cascade — ships with opencv-python, no download)
_face_cascade: cv2.CascadeClassifier | None = None


def _get_face_detector() -> cv2.CascadeClassifier | None:
    """Load OpenCV Haar cascade face detector (bundled with opencv-python)."""
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            logger.warning("Haar cascade loaded but is empty")
            _face_cascade = None
        else:
            logger.info("Face detector loaded (Haar cascade, bundled with OpenCV)")
    except Exception as e:
        logger.warning("Could not load face detector: %s", e)

    return _face_cascade


# ── Frame sampling ─────────────────────────────────────────────────

def _sample_frames(
    video_path: str | Path,
    start: float,
    end: float,
    fps: int = _SAMPLE_FPS,
    width: int = _SAMPLE_W,
    height: int = _SAMPLE_H,
) -> List[np.ndarray]:
    """Extract frames from a video segment at given FPS and resolution."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    start_frame = int(start * video_fps)
    end_frame = int(end * video_fps)
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            resized = cv2.resize(frame, (width, height))
            frames.append(resized)
        frame_idx += 1

    cap.release()
    return frames


# ── Feature extractors ─────────────────────────────────────────────

def _motion_features(frames: List[np.ndarray]) -> Dict[str, float]:
    """Frame differencing + optical flow for motion intensity."""
    if len(frames) < 2:
        return {
            "motion_mean": 0.0,
            "motion_max": 0.0,
            "motion_std": 0.0,
            "optical_flow_mean": 0.0,
            "optical_flow_max": 0.0,
        }

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # Frame differencing
    diffs = []
    for i in range(1, len(grays)):
        diff = cv2.absdiff(grays[i], grays[i - 1])
        diffs.append(float(np.mean(diff)))

    diffs_arr = np.array(diffs)

    # Dense optical flow (Farneback) on subset for speed
    flow_mags = []
    step = max(1, len(grays) // 15)  # sample up to ~15 pairs
    for i in range(step, len(grays), step):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i - step], grays[i],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(float(np.mean(mag)))

    flow_arr = np.array(flow_mags) if flow_mags else np.array([0.0])

    return {
        "motion_mean": float(np.mean(diffs_arr)),
        "motion_max": float(np.max(diffs_arr)),
        "motion_std": float(np.std(diffs_arr)),
        "optical_flow_mean": float(np.mean(flow_arr)),
        "optical_flow_max": float(np.max(flow_arr)),
    }


def _scene_change_features(
    video_path: str | Path,
    start: float,
    end: float,
) -> Dict[str, float]:
    """Detect scene changes using ffmpeg's scene filter."""
    duration = end - start
    cmd = [
        "ffmpeg", "-ss", str(start), "-to", str(end),
        "-i", str(video_path),
        "-vf", "select='gt(scene,0.3)',metadata=print:file=-",
        "-an", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    scores = []
    for line in result.stderr.split("\n"):
        if "scene_score" in line.lower():
            try:
                val = float(line.split("=")[-1].strip())
                scores.append(val)
            except (ValueError, IndexError):
                pass

    # Also parse from stdout (metadata print)
    for line in result.stdout.split("\n"):
        if "lavfi.scene_score" in line:
            try:
                val = float(line.split("=")[-1].strip())
                scores.append(val)
            except (ValueError, IndexError):
                pass

    cut_count = len(scores)
    return {
        "cuts_per_second": cut_count / max(duration, 0.1),
        "scene_change_max": float(max(scores)) if scores else 0.0,
        "scene_change_mean": float(np.mean(scores)) if scores else 0.0,
        "total_cuts": float(cut_count),
    }


def _face_features(frames: List[np.ndarray]) -> Dict[str, float]:
    """Detect faces using OpenCV Haar cascade (bundled, no download needed)."""
    cascade = _get_face_detector()
    if cascade is None or not frames:
        return {
            "face_presence_ratio": 0.0,
            "face_count_mean": 0.0,
            "face_count_max": 0.0,
            "face_area_ratio_mean": 0.0,
        }

    # Sample every other frame for speed
    sample_step = max(1, len(frames) // 20)
    sampled = frames[::sample_step]

    face_counts = []
    face_area_ratios = []
    frame_area = sampled[0].shape[0] * sampled[0].shape[1]

    for frame in sampled:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        count = len(faces)
        total_area = 0.0
        for (x, y, w, h) in faces:
            total_area += w * h

        face_counts.append(count)
        face_area_ratios.append(total_area / frame_area if frame_area > 0 else 0.0)

    counts_arr = np.array(face_counts)
    areas_arr = np.array(face_area_ratios)

    return {
        "face_presence_ratio": float(np.mean(counts_arr > 0)),
        "face_count_mean": float(np.mean(counts_arr)),
        "face_count_max": float(np.max(counts_arr)),
        "face_area_ratio_mean": float(np.mean(areas_arr)),
    }


def _brightness_contrast_features(frames: List[np.ndarray]) -> Dict[str, float]:
    """Brightness and contrast dynamics over time."""
    if not frames:
        return {
            "brightness_mean": 0.0,
            "brightness_std": 0.0,
            "contrast_mean": 0.0,
            "contrast_std": 0.0,
            "brightness_delta_max": 0.0,
        }

    brightness_vals = []
    contrast_vals = []

    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(float(np.mean(gray)))
        contrast_vals.append(float(np.std(gray)))

    b_arr = np.array(brightness_vals)
    c_arr = np.array(contrast_vals)

    # Max brightness change between consecutive frames
    deltas = np.abs(np.diff(b_arr)) if len(b_arr) > 1 else np.array([0.0])

    return {
        "brightness_mean": float(np.mean(b_arr)),
        "brightness_std": float(np.std(b_arr)),
        "contrast_mean": float(np.mean(c_arr)),
        "contrast_std": float(np.std(c_arr)),
        "brightness_delta_max": float(np.max(deltas)),
    }


def _colorfulness_features(frames: List[np.ndarray]) -> Dict[str, float]:
    """Color saturation and Hasler-Suesstrunk colorfulness metric."""
    if not frames:
        return {
            "colorfulness_mean": 0.0,
            "saturation_mean": 0.0,
            "saturation_std": 0.0,
            "hue_entropy": 0.0,
        }

    colorfulness_vals = []
    saturation_vals = []
    hue_histograms = []

    for f in frames:
        # Hasler-Suesstrunk colorfulness
        B, G, R = f[:, :, 0].astype(float), f[:, :, 1].astype(float), f[:, :, 2].astype(float)
        rg = R - G
        yb = 0.5 * (R + G) - B
        std_root = math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        mean_root = math.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        colorfulness_vals.append(std_root + 0.3 * mean_root)

        # HSV saturation
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        saturation_vals.append(float(np.mean(hsv[:, :, 1])))

        # Hue histogram for entropy
        hue_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        hue_hist = hue_hist.flatten() / (hue_hist.sum() + 1e-8)
        hue_histograms.append(hue_hist)

    # Hue entropy (average across frames)
    avg_hue = np.mean(hue_histograms, axis=0)
    avg_hue = avg_hue[avg_hue > 0]
    hue_entropy = -float(np.sum(avg_hue * np.log2(avg_hue + 1e-10)))

    sat_arr = np.array(saturation_vals)

    return {
        "colorfulness_mean": float(np.mean(colorfulness_vals)),
        "saturation_mean": float(np.mean(sat_arr)),
        "saturation_std": float(np.std(sat_arr)),
        "hue_entropy": hue_entropy,
    }


def _visual_complexity_features(frames: List[np.ndarray]) -> Dict[str, float]:
    """Edge density and sharpness (Laplacian variance)."""
    if not frames:
        return {
            "edge_density_mean": 0.0,
            "edge_density_std": 0.0,
            "sharpness_mean": 0.0,
            "sharpness_min": 0.0,
        }

    edge_densities = []
    sharpness_vals = []

    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        # Canny edge ratio
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / (edges.shape[0] * edges.shape[1])
        edge_densities.append(edge_ratio)

        # Laplacian variance (sharpness / blur detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_vals.append(float(laplacian.var()))

    ed_arr = np.array(edge_densities)
    sh_arr = np.array(sharpness_vals)

    return {
        "edge_density_mean": float(np.mean(ed_arr)),
        "edge_density_std": float(np.std(ed_arr)),
        "sharpness_mean": float(np.mean(sh_arr)),
        "sharpness_min": float(np.min(sh_arr)),
    }


def _camera_motion_features(frames: List[np.ndarray]) -> Dict[str, float]:
    """Classify camera motion via affine transform estimation between frames."""
    if len(frames) < 2:
        return {
            "camera_stability": 1.0,
            "zoom_intensity": 0.0,
            "pan_intensity": 0.0,
        }

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    orb = cv2.ORB_create(nfeatures=200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    translations = []
    scales = []
    residuals = []

    step = max(1, len(grays) // 12)
    for i in range(step, len(grays), step):
        kp1, des1 = orb.detectAndCompute(grays[i - step], None)
        kp2, des2 = orb.detectAndCompute(grays[i], None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            continue

        matches = bf.match(des1, des2)
        if len(matches) < 4:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(pts1, pts2)
        if M is None:
            continue

        # Extract translation, scale, rotation from affine matrix
        tx, ty = M[0, 2], M[1, 2]
        sx = math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)

        translations.append(math.sqrt(tx ** 2 + ty ** 2))
        scales.append(abs(sx - 1.0))  # deviation from scale=1

        # Residual = mean reprojection error for inlier points
        if inliers is not None:
            inlier_mask = inliers.flatten().astype(bool)
            if np.any(inlier_mask):
                pts1_in = pts1[inlier_mask]
                transformed = cv2.transform(pts1_in, M)
                err = np.sqrt(np.sum((transformed - pts2[inlier_mask]) ** 2, axis=-1))
                residuals.append(float(np.mean(err)))

    if not translations:
        return {"camera_stability": 1.0, "zoom_intensity": 0.0, "pan_intensity": 0.0}

    trans_arr = np.array(translations)
    scale_arr = np.array(scales)

    # Stability: inverse of mean residual (higher = more stable)
    mean_residual = np.mean(residuals) if residuals else 0.0
    stability = 1.0 / (1.0 + mean_residual)

    return {
        "camera_stability": float(stability),
        "zoom_intensity": float(np.mean(scale_arr)),
        "pan_intensity": float(np.mean(trans_arr)),
    }


# ── Public API ─────────────────────────────────────────────────────

VIDEO_FEATURE_NAMES = [
    # Motion (5)
    "motion_mean",
    "motion_max",
    "motion_std",
    "optical_flow_mean",
    "optical_flow_max",
    # Scene changes (4)
    "cuts_per_second",
    "scene_change_max",
    "scene_change_mean",
    "total_cuts",
    # Faces (4)
    "face_presence_ratio",
    "face_count_mean",
    "face_count_max",
    "face_area_ratio_mean",
    # Brightness/contrast (5)
    "brightness_mean",
    "brightness_std",
    "contrast_mean",
    "contrast_std",
    "brightness_delta_max",
    # Colorfulness (4)
    "colorfulness_mean",
    "saturation_mean",
    "saturation_std",
    "hue_entropy",
    # Visual complexity (4)
    "edge_density_mean",
    "edge_density_std",
    "sharpness_mean",
    "sharpness_min",
    # Camera motion (3)
    "camera_stability",
    "zoom_intensity",
    "pan_intensity",
]


def extract_video_features(
    video_path: str | Path,
    start: float,
    end: float,
) -> Dict[str, float]:
    """Extract all video features for a single segment.

    Returns a dict mapping feature name -> float value.
    """
    features: Dict[str, float] = {}

    # Sample frames once, reuse across extractors
    frames = _sample_frames(video_path, start, end)

    if not frames:
        return {name: 0.0 for name in VIDEO_FEATURE_NAMES}

    # Run all extractors
    features.update(_motion_features(frames))
    features.update(_scene_change_features(video_path, start, end))
    features.update(_face_features(frames))
    features.update(_brightness_contrast_features(frames))
    features.update(_colorfulness_features(frames))
    features.update(_visual_complexity_features(frames))
    features.update(_camera_motion_features(frames))

    return features


def extract_video_features_batch(
    video_path: str | Path,
    candidates: list,
) -> np.ndarray:
    """Extract video features for all candidates.

    Returns (N, len(VIDEO_FEATURE_NAMES)) numpy array.
    """
    rows = []
    total = len(candidates)
    for i, c in enumerate(candidates):
        logger.info("Extracting video features: %d/%d (%.1fs-%.1fs)",
                     i + 1, total, c["start"], c["end"])
        feats = extract_video_features(video_path, c["start"], c["end"])
        row = [feats.get(name, 0.0) for name in VIDEO_FEATURE_NAMES]
        rows.append(row)

    return np.array(rows, dtype=np.float32)
