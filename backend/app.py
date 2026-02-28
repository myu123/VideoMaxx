"""FastAPI application entry point."""

from __future__ import annotations

import logging
import shutil
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import OUTPUTS_DIR
from backend.routers import jobs, labeling, training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _startup_checks() -> dict:
    checks = {}

    # ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    checks["ffmpeg"] = ffmpeg_path is not None
    if ffmpeg_path:
        logger.info("ffmpeg found: %s", ffmpeg_path)
    else:
        logger.error("ffmpeg NOT found on PATH")

    # CUDA
    try:
        import torch
        checks["cuda"] = torch.cuda.is_available()
        if checks["cuda"]:
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("CUDA not available, will use CPU")
    except ImportError:
        checks["cuda"] = False
        logger.info("PyTorch not installed, CUDA check skipped")

    return checks


system_info = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global system_info
    system_info = _startup_checks()
    yield


app = FastAPI(
    title="VideoMaxx — Video Highlight Ranker",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:4000",
        "http://127.0.0.1:4000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve rendered clips as static files
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Routers
app.include_router(jobs.router, prefix="/api")
app.include_router(labeling.router, prefix="/api")
app.include_router(training.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok", **system_info}
