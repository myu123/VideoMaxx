"""Central configuration for VideoMaxx pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
LABELS_CSV = DATA_DIR / "labels.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
ML_ARTIFACTS_DIR = BASE_DIR / "ml_artifacts"

# Ensure dirs exist at import time
for d in [RAW_VIDEOS_DIR, OUTPUTS_DIR, ML_ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Whisper ────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "small"          # "tiny", "small", "medium"
WHISPER_DEVICE = "cuda"               # auto-fallback handled in code
WHISPER_COMPUTE_TYPE = "float16"

# ── Audio ──────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000

# ── Candidate windows ─────────────────────────────────────────────
CANDIDATE_MIN_DURATION = 15.0         # seconds
CANDIDATE_MAX_DURATION = 60.0         # seconds

# ── Text embeddings ────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── ML ─────────────────────────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# ── Clip selection ─────────────────────────────────────────────────
MAX_CLIPS = 5
OVERLAP_IOU_THRESHOLD = 0.3
MIN_START_DISTANCE = 10.0             # seconds

# ── Timeline ───────────────────────────────────────────────────────
TIMELINE_WINDOW = 20.0                # seconds
TIMELINE_STEP = 5.0                   # seconds
