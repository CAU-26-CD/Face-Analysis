"""Demo-only fast path for face analysis.

This module exists solely so the demo build can finish a full analysis in
1-2 seconds instead of the ~30-60s a real job takes. It does that by cutting
the work down to a handful of sampled frames and shrinking the detector input
sizes — accuracy drops accordingly, so this is NOT meant for production runs.

Everything is driven by env vars and gated on ``DEMO_FAST_MODE``. On this
branch the flag defaults to ON, so simply deploying the branch gives you the
fast demo behaviour; set ``DEMO_FAST_MODE=0`` to fall back to the normal
pipeline.

    DEMO_FAST_MODE                 "1"/"0"  (default "1" on this branch)
    DEMO_FRAME_INTERVAL_SECONDS    float    seconds between sampled frames
    DEMO_MAX_FRAMES               int      hard cap on sampled frames total
    DEMO_YOLO_IMGSZ              int      YOLO person-detector input size
    DEMO_FACE_DET_SIZE          int      InsightFace square det_size
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class DemoSettings:
    enabled: bool
    frame_interval_seconds: float
    max_frames: int
    yolo_imgsz: int
    face_det_size: int


@lru_cache(maxsize=1)
def demo_settings() -> DemoSettings:
    """Resolve the demo knobs from the environment once per process."""
    settings = DemoSettings(
        enabled=_env_bool("DEMO_FAST_MODE", default=True),
        frame_interval_seconds=_env_float("DEMO_FRAME_INTERVAL_SECONDS", 1.0),
        max_frames=_env_int("DEMO_MAX_FRAMES", 6),
        yolo_imgsz=_env_int("DEMO_YOLO_IMGSZ", 320),
        face_det_size=_env_int("DEMO_FACE_DET_SIZE", 320),
    )
    if settings.enabled:
        logger.warning(
            "DEMO_FAST_MODE is ON: sampling <=%d frames every %.2fs, "
            "yolo_imgsz=%d face_det_size=%d, thumbnails disabled. "
            "Accuracy is reduced — this build is for demos only.",
            settings.max_frames,
            settings.frame_interval_seconds,
            settings.yolo_imgsz,
            settings.face_det_size,
        )
    return settings
