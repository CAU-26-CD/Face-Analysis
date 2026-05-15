from math import sqrt
from pathlib import Path

from app.services.face_analysis.models import FaceDetection, WithinVideoCluster


class ClusterThumbnailExtractor:
    """Saves a few representative face crops per cluster for human verification.

    For each cluster we want a clear, identifiable face — not the "most typical
    centroid view" (which often lands on hair / occlusion). We split the
    cluster's time range into ``samples_per_cluster`` even buckets and pick the
    highest-quality detection from each: ``confidence * sqrt(bbox_area)``.
    Files are saved as ``<cluster_id>_1.jpg``, ``_2.jpg``, ... in time order.
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
        video_path: str | Path,
        clusters: list[WithinVideoCluster],
        output_dir: str | Path,
    ) -> dict[str, list[Path]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required to crop face thumbnails."
            ) from exc

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        picks_by_frame: dict[int, list[tuple[str, int, FaceDetection]]] = {}
        for cluster in clusters:
            picks = self._pick_representatives(cluster)
            for sample_index, detection in enumerate(picks, start=1):
                picks_by_frame.setdefault(detection.frame_index, []).append(
                    (cluster.cluster_id, sample_index, detection)
                )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        saved: dict[str, list[Path]] = {}
        try:
            for frame_index in sorted(picks_by_frame.keys()):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    continue
                for cluster_id, sample_index, detection in picks_by_frame[frame_index]:
                    crop = self._crop(frame, detection.bbox)
                    if crop is None:
                        continue
                    target = output_path / f"{cluster_id}_{sample_index}.jpg"
                    cv2.imwrite(
                        str(target),
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    saved.setdefault(cluster_id, []).append(target)
        finally:
            capture.release()
        return saved

    def _pick_representatives(self, cluster: WithinVideoCluster) -> list[FaceDetection]:
        detections = [
            detection
            for tracklet in cluster.tracklets
            for detection in tracklet.face_detections
        ]
        if not detections:
            return []

        detections.sort(key=lambda d: d.timestamp_seconds)
        buckets = self._bucket_detections(detections, self.samples_per_cluster)
        picks: list[FaceDetection] = []
        for bucket in buckets:
            if not bucket:
                continue
            picks.append(max(bucket, key=self._quality_score))
        picks.sort(key=lambda d: d.timestamp_seconds)
        return picks

    @staticmethod
    def _bucket_detections(
        detections: list[FaceDetection], bucket_count: int
    ) -> list[list[FaceDetection]]:
        if not detections:
            return []
        start = detections[0].timestamp_seconds
        end = detections[-1].timestamp_seconds
        span = end - start
        if span <= 0 or bucket_count == 1:
            return [list(detections)]
        bucket_size = span / bucket_count
        buckets: list[list[FaceDetection]] = [[] for _ in range(bucket_count)]
        for detection in detections:
            offset = detection.timestamp_seconds - start
            index = min(bucket_count - 1, int(offset / bucket_size))
            buckets[index].append(detection)
        return buckets

    @staticmethod
    def _quality_score(detection: FaceDetection) -> float:
        x1, y1, x2, y2 = detection.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return detection.confidence * sqrt(area)

    def _crop(self, frame, bbox: tuple[float, float, float, float]):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = box_width * self.padding_ratio
        pad_y = box_height * self.padding_ratio
        x1p = max(0, int(round(x1 - pad_x)))
        y1p = max(0, int(round(y1 - pad_y)))
        x2p = min(width, int(round(x2 + pad_x)))
        y2p = min(height, int(round(y2 + pad_y)))
        if x2p <= x1p or y2p <= y1p:
            return None
        return frame[y1p:y2p, x1p:x2p]
