import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.models import KnownActor

logger = logging.getLogger(__name__)

CALLBACK_TIMEOUT_SECONDS = 20.0


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
        analysis = FaceVideoAnalyzer().analyze(
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
        _download_from_s3(bucket_name, s3_key, destination)
        return destination

    _download_from_url(s3_url, destination)
    return destination


def _download_from_s3(bucket_name: str, s3_key: str, destination: Path) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 downloads. Install it with `poetry add boto3`."
        ) from exc

    region = os.getenv("AWS_REGION")
    client_kwargs = {"region_name": region} if region else {}
    client = boto3.client("s3", **client_kwargs)
    client.download_file(bucket_name, s3_key, str(destination))


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
