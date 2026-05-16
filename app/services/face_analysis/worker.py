import logging
import os
import platform
import tempfile
import unicodedata
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.models import KnownActor

logger = logging.getLogger(__name__)

CALLBACK_TIMEOUT_SECONDS = 20.0


def _resolve_accelerator() -> tuple[str | None, list[str] | None]:
    """Resolve (device, onnx_providers) from env or platform.

    FACE_ANALYZER_DEVICE: "mps" | "cuda" | "cpu" (override)
    FACE_ANALYZER_ONNX_PROVIDERS: comma-separated list (override)
    Defaults: Apple Silicon → ("mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"])
              elsewhere → (None, None) so each detector keeps its CPU default.
    """
    device = os.getenv("FACE_ANALYZER_DEVICE")
    providers_env = os.getenv("FACE_ANALYZER_ONNX_PROVIDERS")
    providers = (
        [p.strip() for p in providers_env.split(",") if p.strip()]
        if providers_env
        else None
    )

    if device is None and providers is None:
        is_apple_silicon = (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        )
        if is_apple_silicon:
            device = "mps"
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    return device, providers


def run_analysis_job(request: dict) -> None:
    video_id = int(request["video_id"])
    callback_url = str(request["callback_url"])

    try:
        payload = _analyze_video(request)
    except Exception as exc:
        logger.exception("Face analysis failed for video_id=%s", video_id)
        payload = {
            "video_id": video_id,
            "analysis_status": "failed",
            "error_message": str(exc),
        }

    _post_callback(callback_url, payload)


def _analyze_video(request: dict) -> dict:
    video_id = int(request["video_id"])
    known_actors = _parse_known_actors(request.get("known_actors", []))

    with tempfile.TemporaryDirectory(prefix="face-analyzer-") as temp_dir:
        video_path = _download_video(
            s3_key=str(request["s3_key"]),
            s3_url=str(request["s3_url"]),
            destination_dir=Path(temp_dir),
        )
        thumbnail_dir = Path("/tmp/face_analyzer") / str(video_id)
        device, onnx_providers = _resolve_accelerator()
        logger.info(
            "Accelerator: device=%s onnx_providers=%s",
            device or "cpu",
            onnx_providers or ["CPUExecutionProvider"],
        )
        analysis = FaceVideoAnalyzer(
            device=device,
            onnx_providers=onnx_providers,
        ).analyze(
            video_path,
            known_actors=known_actors,
            thumbnail_dir=thumbnail_dir,
        )
        logger.info("Saved cluster thumbnails to %s", thumbnail_dir)

        return {
            "video_id": video_id,
            "analysis_status": "done",
            "analysis_result": {
                "video_path": analysis.video_path,
                "appearances": [
                    asdict(appearance)
                    for appearance in analysis.appearances
                ],
                "new_candidates": [
                    asdict(candidate)
                    for candidate in analysis.new_candidates
                ],
            },
        }


def _parse_known_actors(raw_actors: list[dict]) -> list[KnownActor]:
    return [
        KnownActor(
            actor_id=str(actor["actor_id"]),
            face_template=[float(value) for value in actor["face_template"]],
        )
        for actor in raw_actors
    ]


def _download_video(s3_key: str, s3_url: str, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(s3_url).path or s3_key).suffix or ".webm"
    destination = destination_dir / f"input{suffix}"

    bucket_name = os.getenv("S3_BUCKET_NAME")
    if bucket_name:
        _download_from_s3_with_normalization_fallback(
            bucket_name, s3_key, destination
        )
        return destination

    _download_from_url(s3_url, destination)
    return destination


def _download_from_s3_with_normalization_fallback(
    bucket_name: str, s3_key: str, destination: Path
) -> None:
    """Try the key as-is, then under the alternate Unicode normalization.

    macOS/iOS upload paths often differ from server-side normalization:
    Hangul filenames in particular are byte-identical visually but encode
    as NFC (precomposed) or NFD (decomposed jamo). HeadObject is byte-
    exact and S3 won't second-guess that, so if BE and the uploader
    disagreed we'd get a phantom 404. Trying the inverse normalization
    on 404 covers both directions without forcing a guess about which
    side is canonical.
    """
    try:
        _download_from_s3(bucket_name, s3_key, destination)
        return
    except Exception as exc:
        if "404" not in str(exc):
            raise
        alt = unicodedata.normalize(
            "NFD" if unicodedata.is_normalized("NFC", s3_key) else "NFC",
            s3_key,
        )
        if alt == s3_key:
            raise
        logger.info(
            "S3 key not found under sent encoding; retrying alternate Unicode form"
        )
        _download_from_s3(bucket_name, alt, destination)


def _download_from_s3(bucket_name: str, s3_key: str, destination: Path) -> None:
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 downloads. Install it with `poetry add boto3`."
        ) from exc

    region = os.getenv("AWS_REGION")
    client_kwargs = {"region_name": region} if region else {}
    client = boto3.client("s3", **client_kwargs)

    max_concurrency = int(os.getenv("S3_DOWNLOAD_MAX_CONCURRENCY", "16"))
    multipart_chunksize = int(
        os.getenv("S3_DOWNLOAD_CHUNKSIZE_MB", "16")
    ) * 1024 * 1024
    config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=multipart_chunksize,
        max_concurrency=max_concurrency,
        use_threads=True,
    )
    client.download_file(bucket_name, s3_key, str(destination), Config=config)


def _download_from_url(s3_url: str, destination: Path) -> None:
    with httpx.stream("GET", s3_url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)


def _post_callback(callback_url: str, payload: dict) -> None:
    secret = os.getenv("ANALYZER_SECRET") or os.getenv("CALLBACK_SECRET")
    if not secret:
        logger.error("ANALYZER_SECRET or CALLBACK_SECRET is required for callback auth")
        return

    try:
        response = httpx.post(
            callback_url,
            json=payload,
            headers={"X-Analyzer-Secret": secret},
            timeout=CALLBACK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception:
        logger.exception("Failed to post face analysis callback to %s", callback_url)
