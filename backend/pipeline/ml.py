"""ML training, scoring, and explainability."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

from backend.config import (
    ML_ARTIFACTS_DIR,
    LABELS_CSV,
    TRAIN_TEST_SPLIT,
    RANDOM_STATE,
)
from backend.pipeline.features import NUMERIC_FEATURE_NAMES, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Human-readable labels for video features in explanations
_FEATURE_EXPLANATIONS: Dict[str, Tuple[str, str]] = {
    # Audio/text
    "speaking_rate":           ("Fast-paced speech", "Slow-paced speech"),
    "rms_mean":                ("High audio energy", "Low audio energy"),
    "rms_max":                 ("Strong audio peaks", "Quiet audio throughout"),
    "spectral_centroid_mean":  ("Bright vocal tone", "Deep vocal tone"),
    "zcr_mean":                ("Dynamic audio texture", "Smooth audio"),
    "word_count":              ("Information-dense segment", "Brief segment"),
    "question_count":          ("Questions engage audience", "No questions"),
    "exclamation_count":       ("Expressive speech", "Neutral delivery"),
    "avg_word_length":         ("Complex vocabulary", "Simple vocabulary"),
    # Motion
    "motion_mean":             ("High visual motion", "Static visuals"),
    "motion_max":              ("Intense motion burst", "No motion spikes"),
    "motion_std":              ("Dynamic motion changes", "Steady motion"),
    "optical_flow_mean":       ("Strong movement flow", "Minimal movement"),
    "optical_flow_max":        ("Peak movement intensity", "No movement peaks"),
    # Scene changes
    "cuts_per_second":         ("Fast-paced editing", "Long continuous shots"),
    "scene_change_max":        ("Major scene transition", "No major transitions"),
    "scene_change_mean":       ("Frequent scene variety", "Consistent scene"),
    "total_cuts":              ("Many cuts (dynamic edit)", "Few cuts (single take)"),
    # Faces
    "face_presence_ratio":     ("Faces on screen", "No faces visible"),
    "face_count_mean":         ("Multiple people visible", "Solo or no faces"),
    "face_count_max":          ("Group shot detected", "Individual framing"),
    "face_area_ratio_mean":    ("Close-up framing", "Wide/distant framing"),
    # Brightness/contrast
    "brightness_mean":         ("Well-lit scene", "Dark scene"),
    "brightness_std":          ("Dynamic lighting", "Flat lighting"),
    "contrast_mean":           ("High visual contrast", "Low contrast"),
    "contrast_std":            ("Contrast variation", "Uniform contrast"),
    "brightness_delta_max":    ("Dramatic light change", "Steady lighting"),
    # Color
    "colorfulness_mean":       ("Vibrant colors", "Muted colors"),
    "saturation_mean":         ("Rich color saturation", "Desaturated visuals"),
    "saturation_std":          ("Color saturation variety", "Uniform saturation"),
    "hue_entropy":             ("Diverse color palette", "Monochromatic"),
    # Visual complexity
    "edge_density_mean":       ("Visually detailed", "Simple/clean visuals"),
    "edge_density_std":        ("Varying visual detail", "Consistent detail level"),
    "sharpness_mean":          ("Sharp/clear image", "Soft/blurry image"),
    "sharpness_min":           ("Consistently sharp", "Some blur detected"),
    # Camera
    "camera_stability":        ("Stable camera work", "Shaky camera"),
    "zoom_intensity":          ("Zoom effects used", "No zoom"),
    "pan_intensity":           ("Camera panning", "Static camera"),
}


def _build_feature_matrix(
    embeddings: np.ndarray,
    numerics: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = False,
) -> Tuple[np.ndarray, StandardScaler]:
    """Concatenate embeddings + scaled numeric features."""
    if fit:
        scaler = StandardScaler()
        numerics_scaled = scaler.fit_transform(numerics)
    else:
        assert scaler is not None
        numerics_scaled = scaler.transform(numerics)

    X = np.hstack([embeddings, numerics_scaled])
    return X, scaler


def train_model(
    job_dirs: List[str | Path],
) -> Dict[str, Any]:
    """Train logistic regression from labeled data across jobs.

    Returns metrics dict.
    """
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df = labels_df[labels_df["label"].isin([0, 1])]

    if len(labels_df) < 5:
        raise ValueError(f"Need at least 5 labeled samples, got {len(labels_df)}")

    all_embeddings = []
    all_numerics = []
    all_labels = []

    for job_dir in job_dirs:
        job_dir = Path(job_dir)
        emb_path = job_dir / "embeddings.npy"
        num_path = job_dir / "numerics.npy"
        cand_path = job_dir / "candidates.csv"

        if not emb_path.exists():
            continue

        embeddings = np.load(emb_path)
        numerics = np.load(num_path)
        candidates = pd.read_csv(cand_path)

        video_id = job_dir.name
        job_labels = labels_df[labels_df["video_id"] == video_id]

        for _, lbl_row in job_labels.iterrows():
            matches = candidates[
                (candidates["start"].round(1) == round(lbl_row["start"], 1)) &
                (candidates["end"].round(1) == round(lbl_row["end"], 1))
            ]
            if len(matches) > 0:
                idx = matches.index[0]
                all_embeddings.append(embeddings[idx])
                all_numerics.append(numerics[idx])
                all_labels.append(int(lbl_row["label"]))

    if len(all_labels) < 5:
        raise ValueError(f"Only {len(all_labels)} matched labeled samples (need >= 5)")

    X_emb = np.array(all_embeddings)
    X_num = np.array(all_numerics)
    y = np.array(all_labels)

    X, scaler = _build_feature_matrix(X_emb, X_num, fit=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y,
    )

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob) if len(set(y_val)) > 1 else 0.0

    top_k = min(5, len(y_val))
    top_indices = np.argsort(y_prob)[-top_k:]
    precision_at_k = float(np.mean(y_val[top_indices]))

    # Save artifacts
    ML_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ML_ARTIFACTS_DIR / "model.joblib")
    joblib.dump(scaler, ML_ARTIFACTS_DIR / "scaler.joblib")

    feature_schema = {
        "embedding_dim": EMBEDDING_DIM,
        "numeric_features": NUMERIC_FEATURE_NAMES,
        "total_features": EMBEDDING_DIM + len(NUMERIC_FEATURE_NAMES),
    }
    with open(ML_ARTIFACTS_DIR / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    metrics = {
        "roc_auc": round(auc, 4),
        "precision_at_5": round(precision_at_k, 4),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "positive_rate": round(float(np.mean(y)), 4),
    }
    with open(ML_ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Model trained — AUC: %.4f, P@5: %.4f", auc, precision_at_k)
    return metrics


def score_candidates(
    embeddings: np.ndarray,
    numerics: np.ndarray,
) -> np.ndarray:
    """Score candidates using trained model. Returns viral_score 0-100."""
    model_path = ML_ARTIFACTS_DIR / "model.joblib"
    scaler_path = ML_ARTIFACTS_DIR / "scaler.joblib"

    if not model_path.exists():
        logger.warning("No trained model found, using heuristic scores")
        return _heuristic_scores(numerics)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X, _ = _build_feature_matrix(embeddings, numerics, scaler=scaler, fit=False)
    probs = model.predict_proba(X)[:, 1]
    scores = (probs * 100).round(1)
    return scores


def _heuristic_scores(numerics: np.ndarray) -> np.ndarray:
    """Heuristic scoring when no model is trained yet.

    Uses weighted combination of features that correlate with engagement.
    """
    if numerics.shape[1] == 0:
        return np.random.uniform(20, 80, size=numerics.shape[0])

    feature_map = {name: i for i, name in enumerate(NUMERIC_FEATURE_NAMES)}

    # Weighted heuristic — video features matter most
    weights = {
        "motion_mean": 3.0,
        "optical_flow_mean": 2.5,
        "cuts_per_second": 2.5,
        "face_presence_ratio": 3.0,
        "face_area_ratio_mean": 2.0,
        "colorfulness_mean": 1.5,
        "contrast_mean": 1.0,
        "brightness_std": 1.5,
        "speaking_rate": 2.0,
        "rms_mean": 2.0,
        "edge_density_mean": 1.0,
        "sharpness_mean": 1.0,
    }

    # Normalize each feature to 0-1
    mins = numerics.min(axis=0)
    maxs = numerics.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (numerics - mins) / ranges

    # Weighted average of available features
    total_weight = 0.0
    weighted_sum = np.zeros(numerics.shape[0])

    for feat_name, weight in weights.items():
        if feat_name in feature_map:
            idx = feature_map[feat_name]
            weighted_sum += normed[:, idx] * weight
            total_weight += weight

    if total_weight > 0:
        raw = weighted_sum / total_weight
    else:
        raw = normed.mean(axis=1)

    # Scale to 10-95 range
    return (raw * 85 + 10).round(1)


def explain_candidate(
    embedding: np.ndarray,
    numerics: np.ndarray,
    text: str,
) -> List[str]:
    """Generate human-readable reasons for a candidate's score."""
    reasons = []

    if numerics.shape[0] == 0:
        return ["No features available"]

    n = numerics[0] if numerics.ndim == 2 else numerics
    feature_map = dict(zip(NUMERIC_FEATURE_NAMES, n))

    # ── Rule-based reasons (thresholds) ─────────────────────────
    # Audio/text
    if feature_map.get("speaking_rate", 0) > 3.0:
        reasons.append("Fast-paced speech")
    if feature_map.get("rms_mean", 0) > 0.05:
        reasons.append("High audio energy")
    if feature_map.get("question_count", 0) >= 2:
        reasons.append("Multiple questions engage audience")
    if feature_map.get("exclamation_count", 0) >= 1:
        reasons.append("Expressive delivery")

    # Video — motion
    if feature_map.get("motion_mean", 0) > 8.0:
        reasons.append("High visual motion")
    elif feature_map.get("optical_flow_mean", 0) > 2.0:
        reasons.append("Strong movement detected")

    # Video — editing
    if feature_map.get("cuts_per_second", 0) > 0.3:
        reasons.append("Fast-paced editing")
    elif feature_map.get("total_cuts", 0) >= 3:
        reasons.append("Multiple scene cuts")

    # Video — faces
    if feature_map.get("face_presence_ratio", 0) > 0.5:
        if feature_map.get("face_area_ratio_mean", 0) > 0.08:
            reasons.append("Close-up face framing")
        else:
            reasons.append("Faces visible on screen")
    if feature_map.get("face_count_max", 0) >= 3:
        reasons.append("Group interaction detected")

    # Video — visuals
    if feature_map.get("colorfulness_mean", 0) > 40:
        reasons.append("Vibrant colors")
    if feature_map.get("brightness_delta_max", 0) > 20:
        reasons.append("Dramatic lighting change")
    if feature_map.get("contrast_mean", 0) > 60:
        reasons.append("High visual contrast")
    if feature_map.get("sharpness_mean", 0) > 500:
        reasons.append("Sharp visual detail")

    # Video — camera
    if feature_map.get("zoom_intensity", 0) > 0.02:
        reasons.append("Zoom effect used")
    if feature_map.get("pan_intensity", 0) > 5.0:
        reasons.append("Dynamic camera movement")

    # ── Trained model coefficient analysis ──────────────────────
    model_path = ML_ARTIFACTS_DIR / "model.joblib"
    if model_path.exists() and len(reasons) < 5:
        model = joblib.load(model_path)
        coefs = model.coef_[0]
        numeric_coefs = coefs[EMBEDDING_DIM:]
        top_idx = np.argsort(np.abs(numeric_coefs))[-3:][::-1]
        for idx in top_idx:
            if idx < len(NUMERIC_FEATURE_NAMES) and len(reasons) < 6:
                feat = NUMERIC_FEATURE_NAMES[idx]
                direction = 0 if numeric_coefs[idx] > 0 else 1
                if feat in _FEATURE_EXPLANATIONS:
                    reasons.append(f"Model: {_FEATURE_EXPLANATIONS[feat][direction]}")

    if not reasons:
        reasons.append("Average engagement signals")

    return reasons[:6]
