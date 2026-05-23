"""RunPod Serverless entrypoint.

Wraps the existing face-analysis pipeline so RunPod can invoke it as a job.
The analyzer is built once at module load so warm workers reuse the loaded
YOLO + InsightFace weights across invocations (cold start pays once).

The BE callback contract is preserved end-to-end: the handler still POSTs the
result to ``callback_url`` with the ``X-Analyzer-Secret`` header via the
existing ``_post_callback`` helper. RunPod's own webhook mechanism is not used
because it cannot attach custom auth headers.
"""

import logging
import os

import runpod

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.worker import _resolve_accelerator, run_analysis_job

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_warm_analyzer() -> FaceVideoAnalyzer:
    device, onnx_providers = _resolve_accelerator()
    logger.info(
        "Warming analyzer: device=%s onnx_providers=%s",
        device or "cpu",
        onnx_providers or ["CPUExecutionProvider"],
    )
    return FaceVideoAnalyzer(device=device, onnx_providers=onnx_providers)


ANALYZER = _build_warm_analyzer()


def handler(job: dict) -> dict:
    """RunPod job handler. ``job["input"]`` carries the BE AnalyzeRequest payload."""
    job_input = job.get("input") or {}
    video_id = job_input.get("video_id")
    logger.info("Received job for video_id=%s", video_id)

    run_analysis_job(job_input, analyzer=ANALYZER)

    # The actual result has already been POSTed to BE via callback_url; the
    # return value here just shows up in the RunPod job log / dashboard.
    return {"video_id": video_id, "delivered_via": "callback_url"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
