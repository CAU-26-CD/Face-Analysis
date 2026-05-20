import logging
import os
import platform
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.models import (
    FaceAnalysisResult,
    KnownActor,
)

logger = logging.getLogger(__name__)

CALLBACK_TIMEOUT_SECONDS = 20.0
DEFAULT_THUMBNAIL_PREFIX = "thumbnails"


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
    requested_thumbnail_dir = request.get("thumbnail_dir")

    with tempfile.TemporaryDirectory(prefix="face-analyzer-") as temp_dir:
        video_path = _download_video(
            s3_key=str(request["s3_key"]),
            s3_url=str(request["s3_url"]),
            destination_dir=Path(temp_dir),
        )
        local_thumbnail_dir = Path(temp_dir) / "thumbnails"
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
            thumbnail_dir=local_thumbnail_dir,
        )

        thumbnail_s3_keys = _upload_cluster_thumbnails(
            analysis.cluster_thumbnail_paths,
            video_id=video_id,
            thumbnail_dir=requested_thumbnail_dir,
        )
        return _build_callback_payload(video_id, analysis, thumbnail_s3_keys)


def _build_callback_payload(
    video_id: int,
    analysis: FaceAnalysisResult,
    thumbnail_s3_keys: dict[str, str],
) -> dict:
    """Translate the analyzer's internal result into the BE callback contract.

    Two ID namespaces collide here: the analyzer uses cluster_id internally
    while BE expects ``"actor:{actor_id}"`` / ``"new:{temp_index}"``. We
    build a cluster_id → temp_index map once for the new-candidate path and
    derive ``actor:{...}`` directly from the (already actor-keyed) appearance
    list for the matched path.
    """
    new_cluster_to_temp_index: dict[str, int] = {
        candidate.cluster_id: index
        for index, candidate in enumerate(analysis.new_candidates)
    }

    matched_payload = [
        {
            "actor_id": m.actor_id,
            "thumbnail_s3_key": thumbnail_s3_keys.get(m.cluster_id),
            "similarity": m.similarity,
            # Novel angles from this video, capped + novelty-filtered
            # by ActorMatcher. BE appends these into the actor's gallery
            # (oldest-drop on overflow).
            "new_exemplars": [list(e) for e in m.new_exemplars],
        }
        for m in analysis.matched
    ]

    new_candidates_payload = [
        {
            "temp_index": index,
            "thumbnail_s3_key": thumbnail_s3_keys.get(candidate.cluster_id),
            # Multi-exemplar seed for BE's new actor row.
            "face_embeddings": [list(e) for e in candidate.embeddings],
            "detection_count": candidate.detection_count,
            "start_seconds": candidate.start_seconds,
            "end_seconds": candidate.end_seconds,
            "suggested_actor_id": candidate.suggested_actor_id,
            "suggested_similarity": candidate.suggested_similarity,
        }
        for index, candidate in enumerate(analysis.new_candidates)
    ]

    # `_build_appearances` keys matched clusters by actor_id and new clusters
    # by cluster_id, so any raw_id we see here is either a temp cluster id
    # (new candidate) or an actor id (matched). Hitting the new-id map first
    # disambiguates without caring about which path the analyzer took.
    appearances_payload = []
    for appearance in analysis.appearances:
        raw_id = appearance.person_id
        if raw_id in new_cluster_to_temp_index:
            person_id = f"new:{new_cluster_to_temp_index[raw_id]}"
        else:
            person_id = f"actor:{raw_id}"
        appearances_payload.append(
            {
                "person_id": person_id,
                "start_seconds": appearance.start_seconds,
                "end_seconds": appearance.end_seconds,
                "detection_count": appearance.detection_count,
            }
        )

    return {
        "video_id": video_id,
        "analysis_status": "done",
        "matched": matched_payload,
        "new_candidates": new_candidates_payload,
        "analysis_result": {
            "video_path": analysis.video_path,
            "appearances": appearances_payload,
        },
    }


def _upload_cluster_thumbnails(
    cluster_thumbnail_paths: dict[str, list[str]],
    video_id: int,
    thumbnail_dir: str | None = None,
) -> dict[str, str]:
    """Upload the top thumbnail per cluster to S3, return cluster_id → key.

    BE-driven layout via ``thumbnail_dir`` (preferred): the caller passes the
    S3 key prefix it wants thumbnails written under — typically
    ``"{project_id}/{session_id}/"`` so video + thumbnails sit side by side
    in one per-session folder. Files land as ``thumb-{idx}.jpg``.

    Legacy fallback (``thumbnail_dir`` is None): keeps the older flat layout
    ``thumbnails/{video_id}/{idx}.jpg`` so callers that haven't been updated
    to send the new field still work.

    Silently no-ops when ``S3_BUCKET_NAME`` is unset (local dev path), so the
    callback can still go out with ``thumbnail_s3_key = None`` for each entry
    rather than failing the whole analysis. Per-cluster upload failures are
    logged and dropped from the map for the same reason.
    """
    if not cluster_thumbnail_paths:
        return {}

    bucket_name = os.getenv("S3_THUMBNAIL_BUCKET") or os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        logger.info(
            "S3_BUCKET_NAME / S3_THUMBNAIL_BUCKET unset; "
            "thumbnails stay local and thumbnail_s3_key will be null"
        )
        return {}

    if thumbnail_dir:
        # Normalize to exactly one trailing slash, regardless of what BE sent.
        normalized_dir = thumbnail_dir.rstrip("/") + "/"
        key_for = lambda idx: f"{normalized_dir}thumb-{idx}.jpg"  # noqa: E731
        log_prefix = normalized_dir
    else:
        legacy_prefix = (
            os.getenv("S3_THUMBNAIL_PREFIX", DEFAULT_THUMBNAIL_PREFIX).strip("/")
            or DEFAULT_THUMBNAIL_PREFIX
        )
        key_for = lambda idx: f"{legacy_prefix}/{video_id}/{idx}.jpg"  # noqa: E731
        log_prefix = f"{legacy_prefix}/{video_id}/"

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 thumbnail uploads. "
            "Install it with `poetry add boto3`."
        ) from exc

    region = os.getenv("AWS_REGION")
    client_kwargs = {"region_name": region} if region else {}
    client = boto3.client("s3", **client_kwargs)

    uploaded: dict[str, str] = {}
    for index, (cluster_id, paths) in enumerate(cluster_thumbnail_paths.items()):
        if not paths:
            continue
        best_path = paths[0]
        s3_key = key_for(index)
        try:
            client.upload_file(
                str(best_path),
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )
        except Exception:
            logger.exception(
                "Failed to upload thumbnail for cluster %s to s3://%s/%s",
                cluster_id,
                bucket_name,
                s3_key,
            )
            continue
        uploaded[cluster_id] = s3_key
    logger.info(
        "Uploaded %d/%d cluster thumbnails to s3://%s/%s",
        len(uploaded),
        len(cluster_thumbnail_paths),
        bucket_name,
        log_prefix,
    )
    return uploaded


def _parse_known_actors(raw_actors: list[dict]) -> list[KnownActor]:
    return [
        KnownActor(
            actor_id=int(actor["actor_id"]),
            face_templates=[
                [float(value) for value in template]
                for template in actor["face_templates"]
            ],
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
