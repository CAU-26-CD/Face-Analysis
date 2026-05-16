from pathlib import Path
from typing import Any

from app.services.face_analysis.exemplars import select_top_k_diverse
from app.services.face_analysis.models import FaceDetection, WithinVideoCluster


def crop_padded(
    frame: Any,
    bbox: tuple[float, float, float, float],
    padding_ratio: float,
) -> Any | None:
    """Slice a padded BGR crop out of ``frame`` for ``bbox``.

    Returns ``None`` when the padded region is degenerate. The returned array
    is a copy so the caller can keep it past the next frame.
    """
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be >= 0")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    x1p = max(0, int(round(x1 - pad_x)))
    y1p = max(0, int(round(y1 - pad_y)))
    x2p = min(width, int(round(x2 + pad_x)))
    y2p = min(height, int(round(y2 + pad_y)))
    if x2p <= x1p or y2p <= y1p:
        return None
    return frame[y1p:y2p, x1p:x2p].copy()


class ClusterThumbnailExtractor:
    """Saves a few representative face crops per cluster for human verification.

    Crops are produced *during* the main analysis loop (the analyzer caches a
    capped per-track top-K of (detection, crop) pairs ranked by quality), so
    this extractor never reopens or seeks the video. That avoids the
    keyframe-seek cost on long rehearsal recordings.

    Picks ``samples_per_cluster`` faces using
    :func:`app.services.face_analysis.exemplars.select_top_k_diverse`, the
    same quality-and-time-diversity selection that identity matching uses,
    so what we *see* lines up with what the matcher *judges by*.
    """

    def __init__(
        self,
        padding_ratio: float = 0.25,
        jpeg_quality: int = 90,
        samples_per_cluster: int = 3,
    ):
        if padding_ratio < 0:
            raise ValueError("padding_ratio must be >= 0")
        if samples_per_cluster < 1:
            raise ValueError("samples_per_cluster must be >= 1")
        self.padding_ratio = padding_ratio
        self.jpeg_quality = jpeg_quality
        self.samples_per_cluster = samples_per_cluster

    def extract(
        self,
        clusters: list[WithinVideoCluster],
        output_dir: str | Path,
        cropped_faces_by_track: dict[str, list[tuple[FaceDetection, Any]]],
    ) -> dict[str, list[Path]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required to encode face thumbnails."
            ) from exc

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved: dict[str, list[Path]] = {}
        for cluster in clusters:
            available: list[tuple[FaceDetection, Any]] = []
            for tracklet in cluster.tracklets:
                available.extend(cropped_faces_by_track.get(tracklet.track_id, []))
            if not available:
                continue

            crop_by_detection_id = {id(detection): crop for detection, crop in available}
            picks = select_top_k_diverse(
                [detection for detection, _ in available],
                self.samples_per_cluster,
            )
            for sample_index, detection in enumerate(picks, start=1):
                crop = crop_by_detection_id.get(id(detection))
                if crop is None:
                    continue
                target = output_path / f"{cluster.cluster_id}_{sample_index}.jpg"
                cv2.imwrite(
                    str(target),
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                saved.setdefault(cluster.cluster_id, []).append(target)
        return saved
